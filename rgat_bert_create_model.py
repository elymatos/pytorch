# =============================================
# 🏗️ Build and Train R-GAT from CSV + BERT
# =============================================
# This script:
# - Loads nodes and edges from CSV
# - Encodes node texts using BERT
# - Trains a Relational GAT (R-GAT)
# - Saves the model and graph state for future use

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ==============================
# 📥 Globals
# ==============================
NEW_NODES = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

# ==============================
# 🔄 Projection from BERT to GNN space
bert_proj = nn.Linear(768, 128).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# =========================
# 📥 Load CSVs
# =========================
nodes_df = pd.read_csv("node_texts.csv")  # columns: node_id,text,label (optional)
edges_df = pd.read_csv("graph_edges.csv")  # columns: source,target,edge_type

# Encode edge types
edge_encoder = LabelEncoder()
edges_df['edge_type_id'] = edge_encoder.fit_transform(edges_df['edge_type'])

# Map node IDs to indices
nodes = pd.Index(edges_df['source'].tolist() + edges_df['target'].tolist()).unique()
node_id_map = {node: i for i, node in enumerate(nodes)}

# Build edge_index and edge_type tensors
source = edges_df['source'].map(node_id_map).tolist()
target = edges_df['target'].map(node_id_map).tolist()
edge_index = torch.tensor([source, target], dtype=torch.long).to(device)
edge_type = torch.tensor(edges_df['edge_type_id'], dtype=torch.long).to(device)

# ==============================
# 🧠 Encode Node Texts with BERT
# ==============================
node_texts = pd.read_csv("node_texts.csv")
node_texts = node_texts.set_index("node_id").reindex(nodes).fillna("")


@torch.no_grad()
def encode_texts(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size].tolist()
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=32)
        outputs = bert(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :])
    return torch.cat(embeddings, dim=0)

x = encode_texts(node_texts['text']).to(device)

# 📦 PyG Data Object
# ==============================
data = Data(x=x, edge_index=edge_index)
data.edge_type = edge_type
x_global = x
nodes_global = list(nodes)
data_global = data

# ==============================
# 🧠 Define R-GAT Layer
# ==============================
class RelationalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, heads=4):
        super().__init__()
        self.rel_embeddings = nn.Embedding(num_relations, in_dim)
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=0.2)

    def forward(self, x, edge_index, edge_type):
        edge_attr = self.rel_embeddings(edge_type)
        return self.gat(x, edge_index, edge_attr)

# ==============================
# 🏗️ Define Full Model
# ==============================
class RGATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.rgat1 = RelationalGATLayer(in_dim, hidden_dim, num_relations)
        self.rgat2 = RelationalGATLayer(hidden_dim, out_dim, num_relations)

    def forward(self, data):
        x = F.relu(self.rgat1(data.x, data.edge_index, data.edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.rgat2(x, data.edge_index, data.edge_type)
        return x

    def embed_nodes(self, data):
        with torch.no_grad():
            x = F.relu(self.rgat1(data.x, data.edge_index, data.edge_type))
        return x

# ==============================
# ⚙️ Setup Model for Inference
# ==============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#labels = torch.randint(0, 3, (data.num_nodes,))  # Replace with actual labels if available
label_encoder = LabelEncoder()
nodes_df['label_id'] = label_encoder.fit_transform(nodes_df['label'])
labels = torch.tensor(nodes_df['label_id'].tolist(), dtype=torch.long)

train_mask = torch.rand(data.num_nodes) < 0.8
data = data.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
num_classes = len(label_encoder.classes_)
model = RGATModel(
    in_dim=x_global.size(1),
    hidden_dim=128,
    out_dim=num_classes,
#    num_relations=len(edge_encoder.classes_)
    num_relations=4
).to(device)
data_global = data_global.to(device)

optimizer = torch.optim.AdamW(list(model.parameters()) + list(bert_proj.parameters()), lr=0.005)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# =========================
# 💾 Save full model and graph
# =========================
torch.save({
    'model': model,
    'bert_proj': bert_proj,
    'edge_encoder': edge_encoder,
    'x': data.x,
    'edge_index': data.edge_index,
    'edge_type': data.edge_type,
    'nodes': nodes_global,
    'label_encoder': label_encoder
}, 'graph_state.pt')

print("✅ Model and graph state saved.")
