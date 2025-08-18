# DGA 域名检测：RandomForest vs XGBoost 对比实验

本子项目基于上级目录的随机森林 Demo，新增 XGBoost 模型，并通过 K 折交叉验证比较二者在同一特征集合上的表现。

## 目标
- 复用特征工程（字符统计 + bigram 概率 + 词典覆盖率）
- 训练 RandomForest 与 XGBoost
- 使用 Stratified K-Fold (默认 k=5) 评估 accuracy / precision / recall / f1 / ROC-AUC
- 输出对比结果并保存最终两个完整模型

## 目录结构
```
DGA_XGBoost对比/
  requirements.txt
  features.py            # 轻量包装，复用主项目特征
  train_compare.py       # 训练 + 交叉验证 + 模型保存
  quick_predict.py       # 快速加载已训练模型预测
```

## 安装依赖
```bash
pip install -r DGA_XGBoost对比/requirements.txt
```

## 运行对比实验
```bash
python DGA_XGBoost对比/train_compare.py compare \
  --csv 随机森林检测DGA/dga_training_data.csv \
  --out DGA_XGBoost对比/model_artifacts \
  --rf-est 150 --xgb-est 300 --k 5
```
输出示例（示意）：
```
Dataset shape: (60000, 12), positive rate=0.500
Performing Stratified K-Fold cross validation (k=5)...

=== CV Average Metrics ===
RandomForest  -> accuracy:0.9985 | precision:0.9983 | recall:0.9986 | f1:0.9984 | roc_auc:0.9995
XGBoost       -> accuracy:0.9989 | precision:0.9990 | recall:0.9987 | f1:0.9988 | roc_auc:0.9996
Artifacts saved to DGA_XGBoost对比/model_artifacts
```
(注：示意值，实际请以终端输出为准。)

## 交叉验证说明
- 使用 StratifiedKFold 维持标签平衡
- 指标取各折平均
- 可通过 `--k` 修改折数，但注意大数据集 + 高折数会增加时间

## 模型训练超参数（默认）
### RandomForest
- n_estimators=150 (可调大提升稳定性)
- class_weight=balanced 以防数据失衡（当前 50/50 实际影响较小）

### XGBoost
- n_estimators=300
- max_depth=6, learning_rate=0.1
- subsample=0.9, colsample_bytree=0.8
- reg_lambda=1.0

> 可进一步网格 / 贝叶斯优化参数，如 max_depth, subsample, colsample_bytree, gamma, min_child_weight, learning_rate。

## 预测单域名
在完成对比并生成模型后：
```bash
python DGA_XGBoost对比/train_compare.py predict --model xgb --domain cloud.gist.build --model-dir DGA_XGBoost对比/model_artifacts
python DGA_XGBoost对比/train_compare.py predict --model rf  --domain 1df5hr42x3s651dgh56tdbq6bs.org --model-dir DGA_XGBoost对比/model_artifacts
```
或使用快速脚本（批量 + 交互）：
```bash
python DGA_XGBoost对比/quick_predict.py --model xgb --model-dir DGA_XGBoost对比/model_artifacts
```

## 结果解读
- 两者都可能在“干净”数据上接近满分，需警惕过拟合与数据简单性
- 更真实的评估应：
  - 引入更噪声的 legit 域名（品牌变体、打字错误）
  - 混入多家族、多算法的 DGA 样本
  - 留出完全独立的外部测试集
  - 关注 PR 曲线（取决于实际环境的基线 DGA 比例）

## 扩展方向
- 添加 LightGBM 做三方对比
- 引入 3-gram / 4-gram 特征或 embedding (char CNN / BiLSTM)
- Online / Streaming 增量更新 (XGBoost 可使用 DMatrix 增量训练思路) 
- SHAP 解释比较两种模型的重要特征差异
- 多模型集成 (stacking / voting)

## 许可证
与主项目一致，仅用于教学与研究；禁止用于恶意目的。
