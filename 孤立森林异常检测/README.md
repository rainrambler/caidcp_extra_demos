# 基于孤立森林的内部威胁 / 异常登录行为检测示例

> 课程模块：AI在安全领域的战术实现  
> 章节：4.2 无监督学习应用——孤立森林 (Isolation Forest) 识别潜在内部威胁  
> 场景定位：在海量认证日志中自动发现“多维组合型”高风险行为（横向移动 / 夜间批量 / 凭证滥用）

---

## 1. 场景与目标

规则/单阈值（如“失败登录>5”）易漏报“多指标协同异常”。孤立森林适合在无标签大规模账号行为中发现稀有行为组合。本示例：  
1. 使用 LANL 认证日志 (auth.txt.gz)  
2. 抽样 → 聚合成“用户-天”特征向量  
3. 训练 IsolationForest → 输出最可疑记录 → 辅助分析师研判

---

## 2. 数据来源与手动下载

原始文件：`auth.txt.gz`（大规模认证事件日志）

下载地址（示例官方链接）：
```
https://csr.lanl.gov/data-fence/1755596898/FPRLQIE1t10IWpb_OeIdK6FKpDA=/cyber1/auth.txt.gz
```
若失效，请访问 LANL Cyber Datasets Portal 或检索 “LANL authentication dataset”。

### 2.1 Windows PowerShell 下载

```powershell
cd 孤立森林异常检测
Invoke-WebRequest -Uri "https://csr.lanl.gov/data-fence/1755596898/FPRLQIE1t10IWpb_OeIdK6FKpDA=/cyber1/auth.txt.gz" -OutFile auth.txt.gz
```

### 2.2 解压

```powershell
gzip -d auth.txt.gz
# 或使用 7-Zip 图形界面；或：
python - <<'PY'
import gzip, shutil
with gzip.open("auth.txt.gz","rb") as fin, open("auth.txt","wb") as fout:
    shutil.copyfileobj(fin,fout)
PY
```

确保结构：
```
孤立森林异常检测/
  if_anomaly_detector.py
  auth.txt
```

### 2.3 体量与采样策略

- 全量极大（数千万行）→ 不直接加载  
- 仅读取前 1,000,000 行 (`nrows=1000000`)  
- 再随机抽 500,000 行做特征聚合  
根据内存可调：`nrows`、`sample(n=...)`

---

## 3. 环境与运行

依赖：`pandas`, `scikit-learn`, `numpy`

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas scikit-learn numpy
python if_anomaly_detector.py
```

---

## 4. 特征工程

聚合粒度：`(source_user, date)`

| 特征 | 说明 | 威胁动机 |
|------|------|----------|
| login_total_count | 当日登录事件总数 | 频率激增 = 自动化 / 滥用 |
| login_night_count | 夜间(22–06)登录总数 | 夜间批量操作可疑度高 |
| dest_computer_nunique | 访问目标主机数量 | 横向移动 / 探测 |
| failed_logins | 失败次数 | 凭证猜测 / 试探 |
| failure_rate | 失败率 | 归一化失败倾向 |

可扩展：时间分布熵、滚动差异、主机多样性指数、成功→失败序列特征等。

---

## 5. 模型

- 算法：`IsolationForest(n_estimators=100, contamination='auto', random_state=42)`
- 输出：
  - `decision_function`：分数（越低越异常）
  - `predict`：-1 异常 / 1 正常
- 适合发现“多维中等偏离的协同异常”而非单值阈值

---

## 6. 运行示例输出（节选）

```
模型在 6662 条用户-天行为记录中，识别出 447 条潜在异常。
异常分数最低（最可疑）Top10: （略）
Top1:
用户: U22@DOM1
login_total_count = 14011
login_night_count = 14011
dest_computer_nunique = 16
failed_logins = 857
failure_rate ≈ 0.061166
anomaly_score = -0.356468
```

全体均值：
```
login_total_count ≈ 75.05
login_night_count ≈ 75.05
dest_computer_nunique ≈ 4.74
failed_logins ≈ 0.52
failure_rate ≈ 0.0133
```

### 6.1 分析

- 全部夜间、大量高频、跨主机、失败显著 → 极端稀有组合
- 可能代表脚本化横向移动 / 凭证批量验证 / 劫持任务
- 孤立森林优势：检测“多维中等偏离的协同异常”而非单指标暴增

---

## 7. 参数与调优

| 场景 | 调整项 | 说明 |
|------|--------|------|
| 降低内存 | 减小 `nrows` 或流式 `chunksize` | 分块累积聚合 |
| 降训练时长 | 降 `n_estimators` | 初学 100 足够 |
| 提升稳定 | 增 `n_estimators` 或多次取交集 | 时间 ↑ |
| 控制告警量 | 显式 `contamination=0.01` | 需结合人工反馈 |
| 可解释性 | 对比用户均值 / 全局均值 | 已实现 Top1 |
| 生产化 | 滚动窗口 & 重训调度 | 处理概念漂移 |

---

## 8. 常见问题

**FutureWarning（pandas 链式赋值）**  
原：
```python
features['failure_rate'].fillna(0, inplace=True)
```
推荐：
```python
features['failure_rate'] = (features['failed_logins'] / features['login_total_count']).fillna(0)
```

**内存不足**：缩小采样或使用：
```python
for chunk in pd.read_csv('auth.txt', header=None, sep=',', chunksize=200000):
    # 逐块清洗 + 追加聚合
```

**噪声过多**：
- 引入对数变换：`features['login_total_count_log'] = np.log1p(features['login_total_count'])`
- 特征标准化后再建模
- 告警合并（同用户相邻多日）

**无标签评估**：
- 构造合成异常（放大特征）
- 与简单规则重合率
- 人工标注小集 → 估计精确率/污染率

---

## 9. 扩展方向

- 序列模型：LSTM / Transformer 捕获跨日时序
- 图表示：用户-主机二部图 → GNN 嵌入 + 异常打分
- 半监督：One-Class SVM / Deep SVDD
- 在线：Kafka/Flink 滚动聚合 + 周期微批训练
- 解释：结合 SHAP（对增强特征模型）

---

## 10. 合规与安全提示

- 遵守数据集使用许可
- 真实日志需脱敏（散列用户 / 主机标识）
- 告警需人工复核再联动处置
- 存储与导出遵循最小权限 & 加密策略

---

## 11. 目录建议

```
孤立森林异常检测/
  if_anomaly_detector.py
  auth.txt         # 大文件（不入库）
  auth.txt.gz      # 原压缩（可选）
  README.md
```

`.gitignore` 建议：
```
孤立森林异常检测/auth.txt
孤立森林异常检测/auth.txt.gz
```

---

## 12. 复现步骤清单

1. 下载 `auth.txt.gz`  
2. 解压得到 `auth.txt`  
3. 安装依赖  
4. 运行脚本：`python if_anomaly_detector.py`  
5. 查看控制台异常 Top10 与解读

---

## 13. 参考

- Liu et al., Isolation Forest (2008)
- scikit-learn 官方文档
- LANL 公共网络安全数据集

---

## 14. 免责声明

代码仅用于教学示例，不保证生产适配。请结合实际安全流程、合规要求与二次验证机制使用。

---
