import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import warnings
import requests
from io import BytesIO
import zipfile
from io import StringIO
import os
import joblib
import logging

# --- 1. 配置区 (Configuration) ---
DATA_CONFIG = {
    "cache_file": 'dga_training_data.csv',
    "legit_url": 'https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip',
    "dga_url": 'https://raw.githubusercontent.com/mitre-attack/attack-dga/main/data/dga_domains.txt',
    "sample_size": 20000
}

MODEL_CONFIG = {
    "model_file": 'dga_rf_model.joblib',
    "random_state": 42,
    "test_size": 0.3
}

# 为超参数调优定义参数网格
# 注意：更大的网格会带来更好的性能，但训练时间也会显著增加
PARAM_GRID = {
    'n_estimators': [100, 200],       # 森林中树的数量
    'max_depth': [10, 20, None],      # 树的最大深度
    'min_samples_leaf': [1, 2],       # 叶子节点最少的样本数
    'min_samples_split': [2, 5]       # 内部节点再划分所需最小样本数
}


FEATURE_NAMES = ['域名长度', '数字比例', '元音比例', '熵', '常见二元组比例']

# --- 2. 功能函数区 (Functions) ---

def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_or_download_data(config):
    """如果本地有缓存则加载，否则从网络下载并保存"""
    cache_file = config["cache_file"]
    if os.path.exists(cache_file):
        logging.info(f"从本地缓存加载数据集: {cache_file}")
        return pd.read_csv(cache_file)
    
    logging.info("--- 正在下载并加载大规模数据集 (首次运行) ---")
    try:
        # 加载合法域名
        req = requests.get(config["legit_url"], timeout=30)
        zip_file = zipfile.ZipFile(BytesIO(req.content))
        with zip_file.open('top-1m.csv') as csv_file:
            csv_content = csv_file.read().decode('utf-8', errors='ignore')
            legit_df = pd.read_csv(StringIO(csv_content), header=None, names=['rank', 'domain'])
        legit_df = legit_df.drop('rank', axis=1)
        legit_df['label'] = 'legit'
        legit_sample = legit_df.head(config["sample_size"])

        # 加载DGA域名
        req = requests.get(config["dga_url"], timeout=30)
        dga_content = req.text
        dga_list = [line.strip() for line in dga_content.strip().split('\n') if line.strip() and not line.startswith('#')]
        dga_df = pd.DataFrame(dga_list, columns=['domain'])
        dga_df['label'] = 'dga'
        dga_sample = dga_df.head(config["sample_size"])

        # 合并并保存
        df = pd.concat([legit_sample, dga_sample], ignore_index=True)
        df = df.sample(frac=1, random_state=MODEL_CONFIG["random_state"]).reset_index(drop=True)
        
        logging.info(f"数据集下载成功！共 {len(df)} 个样本。")
        logging.info(f"正在保存数据集到本地缓存: {cache_file}")
        df.to_csv(cache_file, index=False)
        return df

    except Exception as e:
        logging.error(f"数据下载失败: {e}", exc_info=True)
        exit()

def calculate_entropy(s):
    if not s: return 0
    counts = Counter(s)
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())

def feature_extraction(domain):
    """为单个域名提取特征"""
    domain_name = str(domain).split('.')[0].replace('-', '')
    length = len(domain_name)
    digits = sum(c.isdigit() for c in domain_name)
    vowels = sum(c in 'aeiou' for c in domain_name.lower())
    entropy = calculate_entropy(domain_name)
    
    common_bigrams = {'th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en', 'at', 'ed', 'to', 'or', 'ea', 'hi', 'is', 'ou', 'ar', 'as', 'de', 'rt', 've'}
    bigram_count = 0
    common_bigram_count = 0
    if length > 1:
        for i in range(length - 1):
            if domain_name[i:i+2].lower() in common_bigrams:
                common_bigram_count += 1
        bigram_count = length - 1

    return [
        length,
        digits / length if length > 0 else 0,
        vowels / length if length > 0 else 0,
        entropy,
        common_bigram_count / bigram_count if bigram_count > 0 else 0
    ]

def process_features(df):
    """对整个DataFrame进行特征工程"""
    logging.info("开始进行特征工程...")
    features = df['domain'].apply(feature_extraction).tolist()
    feature_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    logging.info("特征工程完成！")
    return df

def train_and_evaluate_model(df):
    """使用GridSearchCV进行超参数调优，然后训练、评估并保存最优模型"""
    X = df[FEATURE_NAMES]
    y = df['label'].apply(lambda x: 1 if x == 'dga' else 0)

    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        raise ValueError(f"数据集中最少类别的样本数仅为 {min_class_count}，无法进行分层抽样。")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG["test_size"], 
        random_state=MODEL_CONFIG["random_state"], 
        stratify=y
    )

    logging.info("开始进行超参数调优（GridSearchCV），这可能需要较长时间...")
    rf = RandomForestClassifier(random_state=MODEL_CONFIG["random_state"])
    
    # cv=3 表示3折交叉验证
    # verbose=2 会打印出详细的搜索过程信息
    grid_search = GridSearchCV(estimator=rf, param_grid=PARAM_GRID, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    logging.info(f"超参数调优完成！找到的最佳参数为: {grid_search.best_params_}")
    
    # 获取最优模型
    best_model = grid_search.best_estimator_
    
    # 使用最优模型进行评估
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"最优模型在测试集上的准确率: {accuracy:.2%}")
    print("\n--- 最优模型分类报告 ---")
    print(classification_report(y_test, y_pred, target_names=['合法域名 (legit)', 'DGA域名 (dga)']))

    # 保存最优模型
    logging.info(f"正在保存最优模型到: {MODEL_CONFIG['model_file']}")
    joblib.dump(best_model, MODEL_CONFIG['model_file'])
    
    return best_model

def plot_feature_importance(model):
    """绘制并显示特征重要性图"""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        '特征': FEATURE_NAMES,
        '重要性': importances
    }).sort_values(by='重要性', ascending=False)
    
    print("\n--- 特征重要性排序 ---")
    print(feature_importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='特征', data=feature_importance_df, palette='viridis')
    plt.title('DGA域名检测中的特征重要性分析', fontsize=16)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    plt.show()

def predict_new_domains(model, domain_list):
    """使用加载的模型对新域名进行预测"""
    logging.info("对新域名进行预测...")
    features = [feature_extraction(d) for d in domain_list]
    feature_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    predictions = model.predict(feature_df)
    probabilities = model.predict_proba(feature_df)
    
    results = ['DGA域名' if p == 1 else '合法域名' for p in predictions]
    print("\n--- 新域名预测结果 ---")
    for i, domain in enumerate(domain_list):
        dga_prob = probabilities[i][1]
        print(f"域名 '{domain}' 的预测结果是: {results[i]} (判定为DGA的概率: {dga_prob:.2%})")

# --- 3. 主程序入口 (Main Execution) ---
def main():
    """主函数，组织整个流程"""
    setup_logging()
    
    model_file = MODEL_CONFIG['model_file']
    
    if os.path.exists(model_file):
        logging.info(f"从本地加载已训练的最优模型: {model_file}")
        model = joblib.load(model_file)
    else:
        logging.info("未找到已训练的模型，将执行完整的下载、特征工程和超参数调优流程。")
        df = load_or_download_data(DATA_CONFIG)
        df_featured = process_features(df)
        model = train_and_evaluate_model(df_featured)
        plot_feature_importance(model)

    domains_to_predict = ['mysecurebank.com', '8dfg9hsd8fgy3.com', 'download-software.net', 'qwertyasdfgzxcvb.cn']
    predict_new_domains(model, domains_to_predict)

if __name__ == "__main__":
    # 设置Matplotlib以正确显示中文字符
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    main()
