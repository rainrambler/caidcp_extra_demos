# GNN示范：横向移动威胁检测 (Graph Neural Network Demo)

> CAIDCP 课程实验 / Lateral Movement Detection with GNN (PyTorch Geometric)

## 1. 项目简介 (Overview)
本示例展示如何使用 **PyTorch Geometric (PyG)** 与 **GCN (Graph Convolutional Network)** 对经过抽象的企业内部横向移动行为进行风险节点识别。数据来自对公开安全日志（LANL 数据集思想）进行提炼后的一个极小可复现实验子集，通过构建 `用户-计算机` 混合图，将登录/访问事件编码为两跳链：`源计算机 -> 用户 -> 目标计算机`，然后用节点分类方式预测哪些实体（用户或主机）参与了攻击链。

核心目标：快速理解 1) 安全日志 → 图结构转换；2) 基础 GCN 模型搭建与训练；3) 风险评分解释与可视化。

## 2. 数据说明 (Embedded Dataset)
脚本 `gnn_demo.py` 内嵌一个 CSV 字符串，字段：

| 字段 | 含义 | 备注 |
|------|------|------|
| time | 时间顺序ID | 仅作排序示例，无滑动窗口处理 |
| user | 发起行为的账号 | 如 `U722` |
| src  | 行为来源主机 | 如 `C1473` |
| dst  | 行为目标主机 | 如 `C1476` |
| attack | 是否属于已知攻击链(0/1) | 用于生成节点标签 |

标签生成逻辑：凡出现在攻击行(`attack=1`)中的 `user / src / dst` 节点均标记为 1，其余为 0。任务即节点二分类(Node Classification)。

## 3. 图构建机制 (Graph Construction)
函数 `build_graph_data` 关键步骤：
1. 节点集合 = 所有用户 ∪ 所有计算机；使用 `LabelEncoder` 统一映射为连续整数 ID。
2. 节点特征 (x)：简单的 2 维 one-hot：`[1,0]=user`, `[0,1]=computer`。
3. 边 (edge_index)：为体现一次访问事件的两阶段关系，拆成两组定向边并拼接：
   - `src -> user`
   - `user -> dst`
4. 标签 (y)：依据攻击相关实体集合生成 0/1。

这样可使信息在两跳内传播，实现从源端到目标端的上下文聚合。

## 4. 模型结构 (Model)
```
GCN(
  conv1: GCNConv(in_features=2, out=16)
  ReLU + Dropout(0.5)
  conv2: GCNConv(16, 2)
  LogSoftmax -> NLLLoss
)
```
训练：`epochs=200`，优化器 `Adam(lr=0.01, weight_decay=5e-4)`，全图监督训练。

## 5. 执行步骤 (How to Run)
### 5.1 创建并激活环境 (可选)
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

### 5.2 安装依赖
`requirements.txt` 已列出：
```powershell
pip install -r requirements.txt
```
> 注意：`torch` 与 `torch_geometric` 版本需与本机 Python / CUDA 匹配，若安装失败请参考下方“常见问题”。

### 5.3 运行脚本
```powershell
python gnn_demo.py
```
输出包括：
- 步骤日志 (数据→图→训练→预测→可视化)
- 每 20 轮的 Loss
- Top 10 风险节点（含 type 与风险分数）
- 窗口弹出图：节点颜色按风险阈值 ( >0.8 红 / >0.5 橙 / 其余蓝或灰 )

## 6. 结果解释 (Result Interpretation)
`predict_and_analyze` 中对第二类别(logits[:,1])取指数得到风险概率，写入 `node_df.risk_score`。排序输出 Top 10，有助于在更大数据集场景中作为威胁狩猎线索。

可视化 (`visualize_results`) 使用 `networkx + spring_layout`：
- 用户与主机共同绘制为有向二部图展开
- 颜色强调高风险实体，辅助分析潜在横向移动链条

## 7. 常见问题 (FAQ / Troubleshooting)
| 问题 | 现象 | 解决方案 |
|------|------|----------|
| PyTorch Geometric 安装失败 | 报错找不到匹配 wheel | 到 https://pytorch-geometric.readthedocs.io 安装指令生成器；先正确安装匹配版本的 `torch` / `torch_scatter` 等依赖 |
| CUDA 未检测到 | `torch.cuda.is_available() == False` | 若无需 GPU 忽略；或者安装带 CUDA 的 PyTorch Wheel (`pip install torch==...+cu118 -f https://download.pytorch.org/whl/torch_stable.html`) |
| Tk / Matplotlib 后端错误 | 弹出 Tcl/Tk 相关异常 | 安装 tk: `conda install tk -y` 或使用无界面后端：设置环境变量 `MPLBACKEND=Agg` |
| 中文字体乱码 | 图标题中文显示为方块 | 安装中文字体并在代码中添加 `plt.rcParams['font.sans-serif']=['SimHei']`（脚本已内置） |
| Windows 路径或编码问题 | 读取/显示乱码 | 确保脚本文件保存为 UTF-8 (默认)；避免路径包含特殊不可见字符 |

## 8. 扩展实验建议 (Extensions)
1. 特征增强：加入度、最近访问次数、时间差分、角色类型嵌入。
2. 时序建模：将事件按时间窗口切片，采用 Temporal GNN / RGCN。
3. 异构图：分别建用户、主机、进程等多类型节点与多关系边，使用 `HeteroData`。
4. 半监督：仅对少部分已知攻击节点标注，观察泛化能力。
5. 异常检测：改为节点表示学习 + 距离/密度方法 (e.g. LOF) 识别异常。
6. 模型对比：尝试 GraphSAGE、GAT、GIN 并比较性能与稳定性。
7. 解释性：集成 Grad-CAM / Integrated Gradients 或 子图归因 (GNNExplainer)。
8. 数据规模化：替换为真实更大日志 (构造批处理或采样策略)。

## 9. 代码结构与函数映射 (Function Map)
| 函数 | 作用 | 输出要点 |
|------|------|---------|
| `load_and_prepare_data` | 读取内嵌 CSV，编码节点 | `df`, `node_df`, `encoder` |
| `build_graph_data` | 构建特征、边、标签 | `Data(x, edge_index, y)` |
| `GCN` | 两层 GCNConv 分类模型 | 前向返回 `log_softmax` |
| `train_model` | 训练循环 | 每 20 epoch 打印 loss |
| `predict_and_analyze` | 生成风险分数表 | Top 10 风险节点 |
| `visualize_results` | 绘制风险图 | 着色与布局展示 |

## 10. 英文简述 (English Brief)
This demo builds a small User-Host interaction graph from an embedded security log snippet. Each event is decomposed into two directed edges (src->user, user->dst). A simple 2-layer GCN (PyTorch Geometric) performs node classification to highlight entities involved in a labeled attack chain. Output includes top risky nodes and a colored visualization. The project is intentionally minimal to illustrate graph construction, training loop, and interpretation, and can be extended with richer features, heterogeneous graphs, temporal modeling, and explainability techniques.

## 11. 参考 (References)
- PyTorch Geometric Docs: https://pytorch-geometric.readthedocs.io
- Kipf & Welling, ICLR 2017: Semi-Supervised Classification with Graph Convolutional Networks
- 可视化支持：NetworkX, Matplotlib

---
若需集成到更大安全分析平台或扩展请在 issue / 讨论区提出需求。
