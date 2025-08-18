# 随机森林辅助安全应用示例：DGA域名检测

本示例演示如何利用特征工程 + 随机森林(Random Forest)快速构建一个检测可疑 DGA 域名 (Domain Generation Algorithm) 的原型模型。

DGA 被僵尸网络广泛用于动态生成 C2 域名以逃避静态黑名单。通过对域名字符分布/结构进行统计学习，可以在缺乏完整上下文的情况下给出可疑概率，为安全分析前置过滤加速。

## 数据集
- 文件：`dga_training_data.csv`
- 行数：60000（legit 30000 + dga 30000），列：`domain,label`
- label 取值：`legit` 或 `dga`

## 特征工程 (features.py)
对域名（取核心部分，如倒数第二级 label）提取如下特征：
1. length: 字符长度
2. entropy: 香农熵 (字符分布均匀性，高熵更可能是随机 DGA)
3. vowel_count: 元音数量
4. digit_count: 数字数量
5. repeated_char_count: 出现频次>1 的字符总重复次数
6. max_consecutive_digits: 最长连续数字长度
7. max_consecutive_consonants: 最长连续非元音字母长度
8. unique_char_count: 不同字符数
9. vowel_ratio: 元音占比
10. digit_ratio: 数字占比
11. bigram_avg_logp: 在 legit 域名集合上训练出的字符二元组平均对数概率（越低越像随机串）
12. dict_coverage: 在一个内置小型词典中被匹配覆盖的字符比例（越低越像随机串）

> 注：词典与 bigram 模型都极简，仅用于 Demo。实际生产应使用更充足的干净 legit 语料、分词字典以及 TLD 处理、公共后缀列表解析、子域过滤等。

## 训练脚本 (train_dga_rf.py)
提供两个子命令：
- 训练：`python train_dga_rf.py train --csv dga_training_data.csv --out model_artifacts`
- 预测：`python train_dga_rf.py predict --domain exampletest123.com --model-dir model_artifacts`

训练完成后会在输出目录写入：
- `dga_rf_model.joblib`：包含 sklearn RandomForest 模型与特征名称
- `bigram_probs.json`：legit 域名统计得到的 bigram 概率（简易平滑）

## 依赖安装
```
pip install -r requirements.txt
```

`requirements.txt` 内容：`pandas scikit-learn numpy joblib`

## 运行示例
```bash
# 1. 训练
python 随机森林检测DGA/train_dga_rf.py train --csv 随机森林检测DGA/dga_training_data.csv --out 随机森林检测DGA/model_artifacts

# 2. 单域名预测
python 随机森林检测DGA/train_dga_rf.py predict --domain cloud.gist.build --model-dir 随机森林检测DGA/model_artifacts
python 随机森林检测DGA/train_dga_rf.py predict --domain 1df5hr42x3s651dgh56tdbq6bs.org --model-dir 随机森林检测DGA/model_artifacts

# 3. 快速批量测试
python 随机森林检测DGA/quick_test.py --model-dir 随机森林检测DGA/model_artifacts
```

输出示例：
```bash
=== Evaluation Metrics ===
accuracy: 0.98xx
precision: 0.98xx
recall: 0.98xx
f1: 0.98xx
roc_auc: 0.99xx
...
Domain: 1df5hr42x3s651dgh56tdbq6bs.org
Predicted label: dga
Probability DGA: 0.9973
Top feature signals (heuristic):
  entropy: ...
  bigram_avg_logp: ...
  vowel_ratio: ...
  ...

1df5hr42x3s651dgh56tdbq6bs.org           -> dga   (p_dga=1.0000)
675wwi1hb3y9w1griggr1vxpg33.net          -> dga   (p_dga=1.0000)
cloud.gist.build                         -> legit (p_dga=0.0000)
knotch.it                                -> legit (p_dga=0.0000)
auth.example.com                         -> legit (p_dga=0.0000)

Enter domains line-by-line (empty line to exit):
> www.csa-c.cn
www.csa-c.cn                             -> legit (p_dga=0.0000)
> asdklfjasl234kj232j4l2kjkl.com     
asdklfjasl234kj232j4l2kjkl.com           -> dga   (p_dga=0.7200)
```
(实际数值视随机种子略有差异。)

## 设计说明
- 使用 RandomForest 原因：鲁棒、可解释性较好（可查看特征重要性）、对特征缩放不敏感、实现简单。
- bigram_avg_logp 有助于捕获合法域名字符转移模式；DGA 往往表现为低概率组合。
- dict_coverage 让模型识别可读词片段，提升对正常品牌/业务域名的识别。
- 可扩展：加入 TLD 类别、子域深度、3-gram、马尔科夫模型、字符类别转移计数、Alexa/toplist rank、WHOIS / DNS 查询 (需要在线数据) 等。

## 局限与改进
1. 词典规模与语料有限——真实环境需更全面的合法域名库。
2. 未处理 punycode / IDN 域名。
3. 没有时间序列、解析结果 (A/NS/MX) 与流量行为特征。
4. 阈值简单取 0.5，可根据 ROC/PR 调优以控制 FP/FN。
5. 未做特征漂移监控，需要定期再训练。

## 下一步可扩展建议
- 使用 `LightGBM` / `XGBoost` 对比性能。
- 引入 `n-gram language model`（3-gram/4-gram）概率向量。
- 构建一个批量预测/REST API 服务端。
- 引入 Explainable AI（SHAP / LIME）展示单样本解释。
- 加入模型版本化与评估报告自动生成。

## 许可
本示例仅用于教学与研究目的，禁止用于绕过安全监测或任何恶意用途。
