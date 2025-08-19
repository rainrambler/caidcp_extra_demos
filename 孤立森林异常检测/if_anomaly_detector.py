# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
# 课程模块: AI在安全领域的战术实现
# 章节: 4.2 - 无监督学习应用：基于孤立森林的内部威胁检测
# MVP示例代码
# -------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

# --- 步骤 1: 数据获取与加载 ---
# 设计思路：首先，我们需要加载数据。真实环境中，这可能来自SIEM、EDR或日志管理平台。
# 在本示例中，我们直接从URL加载LANL认证数据集。

# 如果本地存在auth.txt.gz文件，则优先使用本地文件
# try:
#     df = pd.read_csv('auth.txt.gz', compression='gzip', header=None, sep=',')
#     print("步骤 1: 成功加载本地文件 auth.txt.gz")
# except FileNotFoundError:
#     print("步骤 1: 本地文件 auth.txt.gz 不存在，开始下载...")
#     # 数据集较大，下载可能需要一些时间
# try:
#     # 尝试从官方源下载
#     url = 'https://csr.lanl.gov/data-fence/1755596898/FPRLQIE1t10IWpb_OeIdK6FKpDA=/cyber1/auth.txt.gz'
#     df = pd.read_csv(url, compression='gzip', header=None, sep=',')
# except Exception as e:
#     # 如果官方源下载失败，提供备用方案或提示
#     print(f"从官方源下载失败: {e}")
#     print("请手动下载 auth.txt.gz 文件并放置在代码同目录下。")
#     # 假设文件已手动下载
#     # df = pd.read_csv('auth.txt.gz', compression='gzip', header=None, sep=',')

# 战术一：采样与迭代处理 - A) 精确采样加载
# 我们不再加载整个文件，而是仅加载前100万行作为样本进行分析。
# 这足以在几秒钟内完成，并验证整个处理流程的正确性。

print("战术一：开始使用采样加载策略...")
# 解压后的文件名为 auth.txt
file_path = 'auth.txt' 
# 注意：确保 auth.txt 文件与python脚本在同一目录下

try:
    df = pd.read_csv(
        file_path, 
        header=None, 
        sep=',',
        nrows=1000000  # <--- 核心改动：仅读取前100万行
    )
    df.columns = ['time', 'source_user', 'dest_user', 'source_computer', 'dest_computer', 
                  'auth_type', 'logon_type', 'auth_orientation', 'success_fail']
    print("通过采样策略，成功加载了100万行数据样本。")
    print("数据加载完成。")
    # ... 后续代码可以继续执行 ...

except FileNotFoundError:
    print(f"错误：请确保解压后的 '{file_path}' 文件与脚本位于同一目录。")
    exit(1)


print("原始数据预览:")
print(df.head())
print("\n" + "="*50 + "\n")


# --- 步骤 2: 数据预处理与特征工程 ---
# 设计思路：原始日志是离散事件，无法直接用于模型。我们需要将其聚合，构建出您所描述的
# “每个员工每天的行为特征向量”。这是将安全事件转化为机器学习语言的关键一步。

print("步骤 2: 开始数据预处理与特征工程...")

# 时间戳转换为日期，方便按天聚合
df['date'] = pd.to_datetime(df['time'], unit='s').dt.date

# 为了教学目的和计算效率，我们仅抽取部分数据进行分析
# 在真实场景中，可能会使用更长的时间窗口或分布式计算
df_sample = df.sample(n=500000, random_state=42) # 随机采样50万条记录

# 创建特征
# 核心思想：围绕“谁（source_user）”在“哪天（date）”做了什么来进行聚合
# 我们将模拟您提出的特征向量: [登录总次数, 夜间登录次数, 访问敏感服务器数量, ...]

# 定义夜间时间（例如，晚上10点到早上6点）
df_sample['hour'] = pd.to_datetime(df_sample['time'], unit='s').dt.hour
df_sample['is_night'] = ((df_sample['hour'] >= 22) | (df_sample['hour'] <= 6)).astype(int)

# 按用户和日期进行分组聚合
features = df_sample.groupby(['source_user', 'date']).agg(
    # 特征1: 登录总次数
    login_total_count = pd.NamedAgg(column='time', aggfunc='count'),
    
    # 特征2: 夜间登录次数
    login_night_count = pd.NamedAgg(column='is_night', aggfunc='sum'),
    
    # 特征3: 访问目标计算机数量 (替代“访问敏感服务器数量”)
    # 逻辑：一个用户一天内访问的目标主机越多，行为越可能异常
    dest_computer_nunique = pd.NamedAgg(column='dest_computer', aggfunc='nunique'),
    
    # 特征4: 登录失败次数
    # 逻辑：大量的失败尝试可能是凭证猜测或账户滥用的迹象
    failed_logins = pd.NamedAgg(column='success_fail', aggfunc=lambda x: (x == 'Fail').sum())
).reset_index()

# 派生新特征：失败率（避免链式赋值 FutureWarning）
features['failure_rate'] = (features['failed_logins'] / features['login_total_count']).fillna(0)
# 可选：对数变换示例（如需增强鲁棒性，可解除注释）
# features['login_total_count_log'] = np.log1p(features['login_total_count'])

print("特征工程完成。")
print("生成的特征向量预览:")
print(features.head())
print("\n" + "="*50 + "\n")


# --- 步骤 3: 模型训练 ---
# 设计思路：应用孤立森林算法。该算法无需标签，非常适合发现“未知的未知”异常。
# contamination参数是关键，它代表我们预估数据中异常点的比例。这通常需要根据经验设定。

print("步骤 3: 开始训练孤立森林模型...")

# 准备用于模型训练的数据（只使用数值特征）
X = features.drop(columns=['source_user', 'date'])

# 初始化并训练模型
# contamination='auto' 是一个较好的起点，也可以设为具体值如 0.01 (1%)
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X)

print("模型训练完成。")
print("\n" + "="*50 + "\n")


# --- 步骤 4: 异常检测与结果分析 ---
# 设计思路：模型的输出是-1（异常）或1（正常），以及一个异常分数。
# 我们需要将这些技术输出翻译成对安全分析师有价值的洞察。

print("步骤 4: 开始进行异常检测并分析结果...")

# 进行预测，获取异常分数和标签
features['anomaly_score'] = iso_forest.decision_function(X)
features['is_anomaly'] = iso_forest.predict(X) # -1 表示异常, 1 表示正常

# 筛选出被模型标记为异常的记录
anomalies = features[features['is_anomaly'] == -1].sort_values(by='anomaly_score')

print(f"模型在 {len(features)} 条用户-天行为记录中，识别出 {len(anomalies)} 条潜在异常。")
print("异常分数最低（最可疑）的TOP 10行为记录:")
print(anomalies.head(10))
print("\n" + "="*50 + "\n")

# --- 教学化结果解读 ---
# 设计思路：这是最关键的一步，将AI的发现与安全场景联系起来。
# 我们需要解释为什么这些行为被认为是异常的。

print("教学化结果解读:")
if not anomalies.empty:
    top_anomaly = anomalies.iloc[0]
    user = top_anomaly['source_user']
    date = top_anomaly['date']
    
    print(f"\n以最异常的记录为例进行分析：")
    print(f"用户: {user}")
    print(f"日期: {date}")
    print(f"该日的行为向量: {top_anomaly.to_dict()}")
    
    # 与该用户的平均行为进行对比，提供上下文
    user_avg_behavior = features[features['source_user'] == user][X.columns].mean()
    print(f"\n该用户的平均行为:")
    print(user_avg_behavior)
    
    # 与全体用户的平均行为进行对比
    overall_avg_behavior = features[X.columns].mean()
    print(f"\n全体用户的平均行为:")
    print(overall_avg_behavior)

    print("\n分析结论:")
    print("安全分析师应重点关注此告警。模型将此记录标记为高度异常，可能是因为其特征组合（例如，极高的登录次数、访问了远超平常数量的机器、同时伴有多次夜间登录和失败尝试）在整体数据分布中极为罕见。")
    print("这正是孤立森林的威力：它并非基于单一阈值（如'失败登录 > 5次'），而是识别出多个指标的异常组合，这种组合可能预示着复杂的、隐蔽的攻击行为，如横向移动或数据窃取。")
else:
    print("在此次抽样数据中未发现显著异常。")