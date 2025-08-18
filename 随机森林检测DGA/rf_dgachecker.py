import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# --- 环境设置 ---
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 步骤 1: 数据加载 (增加本地缓存功能) ---
COMBINED_DATA_FILE = 'dga_training_data.csv'

if os.path.exists(COMBINED_DATA_FILE):
    print(f"从本地缓存加载数据集: {COMBINED_DATA_FILE}")
    df = pd.read_csv(COMBINED_DATA_FILE)
else:
    print("--- 正在下载并加载大规模数据集 (首次运行) ---")
    print("这可能需要一点时间，请稍候...")
    try:
        # 加载合法域名 (Cisco Umbrella Top 1 Million)
        legit_url = 'https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip'
        req = requests.get(legit_url, timeout=30)
        zip_file = zipfile.ZipFile(BytesIO(req.content))
        with zip_file.open('top-1m.csv') as csv_file:
            csv_content = csv_file.read().decode('utf-8', errors='ignore')
            legit_df = pd.read_csv(StringIO(csv_content), header=None, names=['rank', 'domain'])
        legit_df = legit_df.drop('rank', axis=1)
        legit_df['label'] = 'legit'
        legit_sample = legit_df.head(30000)

        # 加载DGA域名 (来自另一个稳定可靠的数据集 - 已更新URL)
        dga_url = 'https://raw.githubusercontent.com/sph116/DGA_domain_test/refs/heads/master/zeus_dga_domains.txt' # <-- 已更新为更可靠的URL
        req = requests.get(dga_url, timeout=30)
        dga_content = req.text
        # 过滤掉空行
        dga_list = [line.strip() for line in dga_content.strip().split('\n') if line.strip()]
        dga_df = pd.DataFrame(dga_list, columns=['domain'])
        dga_df['label'] = 'dga'
        dga_sample = dga_df.head(30000)

        # 合并数据集
        df = pd.concat([legit_sample, dga_sample], ignore_index=True)
        # 打乱数据集顺序
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"数据集下载成功！共 {len(df)} 个样本。")
        print(f"正在保存数据集到本地缓存: {COMBINED_DATA_FILE}")
        df.to_csv(COMBINED_DATA_FILE, index=False)

    except Exception as e:
        print(f"数据下载失败: {e}")
        print("无法继续，请检查网络连接或URL是否有效。")
        exit() # 如果下载失败，则退出程序

# --- 数据清洗与验证 ---
df.dropna(subset=['domain'], inplace=True)
df['domain'] = df['domain'].astype(str)

print("\n--- 数据集类别分布 ---")
print(df['label'].value_counts())

# 检查每个类别是否至少有2个样本，以满足分层抽样的要求
min_class_count = df['label'].value_counts().min()
if min_class_count < 2:
    raise ValueError(f"数据集中最少类别的样本数仅为 {min_class_count}，无法进行分层抽样。请检查数据源。")

# --- 步骤 2: 特征工程 ---
def calculate_entropy(s):
    if not s: return 0
    counts = Counter(s)
    length = len(s)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def feature_extraction(domain):
    domain_name = str(domain).split('.')[0].replace('-', '')
    length = len(domain_name)
    digits = sum(c.isdigit() for c in domain_name)
    digit_ratio = digits / length if length > 0 else 0
    vowels = sum(c in 'aeiou' for c in domain_name.lower())
    vowel_ratio = vowels / length if length > 0 else 0
    entropy = calculate_entropy(domain_name)
    common_bigrams = {'th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en', 'at', 'ed', 'to', 'or', 'ea', 'hi', 'is', 'ou', 'ar', 'as', 'de', 'rt', 've'}
    bigram_count = 0
    common_bigram_count = 0
    if length > 1:
        for i in range(length - 1):
            bigram = domain_name[i:i+2].lower()
            bigram_count += 1
            if bigram in common_bigrams:
                common_bigram_count += 1
    common_bigram_ratio = common_bigram_count / bigram_count if bigram_count > 0 else 0
    return [length, digit_ratio, vowel_ratio, entropy, common_bigram_ratio]

print("\n--- 开始进行特征工程 ---")
feature_names = ['域名长度', '数字比例', '元音比例', '熵', '常见二元组比例']
features = df['domain'].apply(feature_extraction).tolist()
feature_df = pd.DataFrame(features, columns=feature_names)
df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
print("特征工程完成！")

# --- 步骤 3: 模型训练 ---
X = df[feature_names]
y = df['label'].apply(lambda x: 1 if x == 'dga' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("\n--- 开始训练最终的随机森林模型 ---")
rf_model.fit(X_train, y_train)
print("模型训练完成！")

# --- 步骤 4: 模型评估与分析 ---
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- 模型评估 (最终版) ---")
print(f"模型在测试集上的准确率: {accuracy:.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['合法域名 (legit)', 'DGA域名 (dga)']))

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': importances
}).sort_values(by='重要性', ascending=False)

print("\n--- 特征重要性排序 (最终版) ---")
print(feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance_df, palette='viridis')
plt.title('DGA域名检测中的特征重要性分析 (最终版)', fontsize=16)
plt.xlabel('重要性得分', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()
plt.show()

# --- 步骤 5: 应用最终模型进行新域名预测 ---
def predict_dga(domain_list, model, feature_names):
    features = [feature_extraction(d) for d in domain_list]
    feature_df = pd.DataFrame(features, columns=feature_names)
    predictions = model.predict(feature_df)
    probabilities = model.predict_proba(feature_df)
    
    results = ['DGA域名' if p == 1 else '合法域名' for p in predictions]
    for i, domain in enumerate(domain_list):
        dga_prob = probabilities[i][1]
        print(f"域名 '{domain}' 的预测结果是: {results[i]} (判定为DGA的概率: {dga_prob:.2%})")

new_domains = ['mysecurebank.com', '8dfg9hsd8fgy3.com', 'download-software.net', 'qwertyasdfgzxcvb.cn']
print("\n--- 对新域名进行预测 (使用最终模型) ---")
predict_dga(new_domains, rf_model, feature_names)
