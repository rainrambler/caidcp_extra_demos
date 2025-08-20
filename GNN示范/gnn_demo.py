# -*- coding: utf-8 -*-
# CAIDCP 课程实验：基于 PyTorch Geometric 的 GNN 横向移动威胁检测 (V2 - 已修正)

# --- 1. 导入必要的库 ---
# 确保您已经按照实验手册的指引，安装了所有依赖库
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

# --- 2. 实验数据 ---
# 我们将使用一个经过简化的、源自LANL开源数据集的子集。
# 为方便实验，数据已直接嵌入代码中。
# 字段说明: time(时间点), user(源用户), src(源计算机), dst(目标计算机), attack(是否为已知攻击环节)
csv_data = """time,user,src,dst,attack
1,U2,C1,C2,0
2,U4,C3,C4,0
3,U722,C1473,C1474,1
4,U722,C1473,C1475,1
5,U722,C1473,C1476,1
6,U6,C5,C6,0
7,U722,C1476,C1025,1
8,U9,C7,C8,0
9,U722,C1025,C21,1
10,U12,C9,C10,0
11,U15,C11,C12,0
12,U722,C1473,C1477,1
13,U20,C13,C14,0
14,U22,C15,C16,0
15,U722,C1473,C1478,1
"""

def load_and_prepare_data():
    """
    加载并预处理数据。
    核心任务：将字符串ID（如'U722', 'C1473'）映射为从0开始的整数ID。
    """
    print("[步骤 1] 正在加载和预处理数据...")
    df = pd.read_csv(StringIO(csv_data))

    # 提取所有不重复的用户和计算机ID
    users = pd.concat([df['user']]).unique()
    computers = pd.concat([df['src'], df['dst']]).unique()
    
    # 将所有实体（用户+计算机）放入一个列表中，并进行编码
    all_nodes = list(users) + list(computers)
    node_encoder = LabelEncoder()
    node_encoder.fit(all_nodes)

    # 创建一个DataFrame来存储节点信息
    node_df = pd.DataFrame({'id': all_nodes})
    node_df['mapped_id'] = node_encoder.transform(all_nodes)
    node_df['type'] = ['user'] * len(users) + ['computer'] * len(computers)
    
    print(f"数据加载完成。共发现 {len(users)} 个用户, {len(computers)} 台计算机。")
    return df, node_df, node_encoder

def build_graph_data(log_df, node_df, node_encoder):
    """
    将日志数据转换为PyTorch Geometric的图数据对象。
    """
    print("\n[步骤 2] 正在从日志构建图谱...")
    
    # --- 节点特征 (Node Features) ---
    # 我们使用一个简单的独热编码来表示节点类型：[is_user, is_computer]
    user_feature = [1, 0]
    computer_feature = [0, 1]
    
    node_features = []
    for _, row in node_df.sort_values('mapped_id').iterrows():
        if row['type'] == 'user':
            node_features.append(user_feature)
        else:
            node_features.append(computer_feature)
    
    x = torch.tensor(node_features, dtype=torch.float)

    # --- 边索引 (Edge Index) ---
    # 【关键修正】我们构建两种关系: src -> user 和 user -> dst 来形成完整的事件链
    src_nodes_mapped = node_encoder.transform(log_df['src'])  # numpy ndarray
    user_nodes_mapped = node_encoder.transform(log_df['user'])
    dst_nodes_mapped = node_encoder.transform(log_df['dst'])

    # 边类型1: 源计算机 -> 用户  / 边类型2: 用户 -> 目标计算机
    # 避免 PyTorch 关于 "Creating a tensor from a list of numpy.ndarrays is extremely slow" 的警告，
    # 先使用 numpy 堆叠为单一 ndarray，再一次性转换为 tensor（共享内存，效率更高）。
    edges_from_src = torch.from_numpy(np.vstack((src_nodes_mapped, user_nodes_mapped))).long()
    edges_to_dst = torch.from_numpy(np.vstack((user_nodes_mapped, dst_nodes_mapped))).long()
    
    # 合并两种边，构建完整的图连接关系
    edge_index = torch.cat([edges_from_src, edges_to_dst], dim=1)

    # --- 标签 (Labels) ---
    # 标签逻辑保持不变：与攻击相关的节点为1，其他为0
    attack_users = log_df[log_df['attack'] == 1]['user'].unique()
    attack_comps_src = log_df[log_df['attack'] == 1]['src'].unique()
    attack_comps_dst = log_df[log_df['attack'] == 1]['dst'].unique()
    attack_nodes_set = set(attack_users) | set(attack_comps_src) | set(attack_comps_dst)
    
    labels = []
    node_df_sorted = node_df.sort_values('mapped_id')
    for _, row in node_df_sorted.iterrows():
        if row['id'] in attack_nodes_set:
            labels.append(1)
        else:
            labels.append(0)
            
    y = torch.tensor(labels, dtype=torch.long)
    
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    print("图谱构建完成，已包含完整的事件链。")
    return graph_data

# --- 3. 定义GNN模型 ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_model(model, data, epochs=200):
    print("\n[步骤 3] 正在训练GNN模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    print("模型训练完成。")

def predict_and_analyze(model, data, node_df):
    print("\n[步骤 4] 正在使用模型进行风险预测与分析...")
    model.eval()
    with torch.no_grad():
        pred_log_softmax = model(data)
    
    pred_prob = torch.exp(pred_log_softmax)[:, 1]
    
    node_df['risk_score'] = pred_prob.cpu().numpy()
    
    top_10_risky_nodes = node_df.sort_values('risk_score', ascending=False).head(10)

    print("\n--- GNN分析结果：Top 10 高风险节点 ---")
    print(top_10_risky_nodes[['id', 'type', 'risk_score']].to_string(index=False))
    print("------------------------------------")
    
    return node_df

def visualize_results(node_df, edge_df):
    print("\n[步骤 5] 正在生成结果可视化图...")
    G = nx.DiGraph()
    
    for _, row in node_df.iterrows():
        color = 'skyblue' if row['type'] == 'user' else 'lightgray'
        if row['risk_score'] > 0.8: color = 'red'
        elif row['risk_score'] > 0.5: color = 'orange'
        G.add_node(row['id'], type=row['type'], color=color)

    # 【可视化修正】我们需要绘制完整的 src -> user -> dst 边
    for _, row in edge_df.iterrows():
        G.add_edge(row['src'], row['user'])
        G.add_edge(row['user'], row['dst'])

    # always set font so that Chinese characters are displayed correctly
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, arrowsize=15, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_color='black')
    
    plt.title("GNN 威胁检测结果可视化 (红色/橙色为高风险节点)", fontsize=18)
    plt.axis('off')
    print("可视化窗口已生成，请查看。")
    plt.show()

# --- 主函数 ---
if __name__ == '__main__':
    log_df, node_df, node_encoder = load_and_prepare_data()
    graph_data = build_graph_data(log_df, node_df, node_encoder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=graph_data.num_node_features, num_classes=2).to(device)
    data = graph_data.to(device)
    
    train_model(model, data)
    final_node_df = predict_and_analyze(model, data, node_df)
    visualize_results(final_node_df, log_df)
