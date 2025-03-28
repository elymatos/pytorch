# ============================================
# 🔁 R-GAT Prototype with BERT + Edge Types
# ============================================
# Features:
# - Loads graph from CSV file
# - Encodes nodes using BERT
# - Uses custom R-GAT layer with edge-type-aware attention
# - Supports model saving and loading
# - Relation types included in edge attributes and training logic
# - Includes link prediction and dynamic graph update

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
import json

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

# ==============================
# 🔁 Load Graph State If Available
# ==============================
if os.path.exists('graph_state.pt'):
    print("🔁 Loading saved graph state from disk...")
    saved = torch.load('graph_state.pt')
    x = saved['x']
    edge_index = saved['edge_index']
    edge_type = saved['edge_type']
    nodes = saved['nodes']
    NEW_NODES = saved.get('new_nodes', {})
else:
    # ==============================
    # 🧾 Load Graph from CSV
    # ==============================
    edges_df = pd.read_csv("graph_edges.csv")

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

# ==============================
# 📦 PyG Data Object
# ==============================
data = Data(x=x, edge_index=edge_index)
data.edge_type = edge_type
# ==============================

# ==============================
# 🔗 Dynamic Insertion + Prediction
# ==============================
# Globals to track modifiable variables
# Ensure they are properly mutable (not reassigned in function scope)
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
model = RGATModel(
    in_dim=x_global.size(1),
    hidden_dim=128,
    out_dim=3,
#    num_relations=len(edge_encoder.classes_)
    num_relations=4
).to(device)
data_global = data_global.to(device)

# Load model if saved
if os.path.exists('rgat_model.pt'):
    print("📦 Loading model weights from rgat_model.pt...")
    checkpoint = torch.load('rgat_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'bert_proj_state_dict' in checkpoint:
        bert_proj.load_state_dict(checkpoint['bert_proj_state_dict'])
    model.eval()
else:
    print("⚠️ Warning: No saved model found. Using untrained model.")



# ==============================
@torch.no_grad()
def predict_and_insert_node(text, node_name=None, top_k=5):
    global x_global, nodes_global, data_global
    model.eval()
    inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=32)
    new_embed = bert_proj(bert(**inputs).last_hidden_state[:, 0, :].to(device))

    if node_name is None:
        node_name = f"new_node_{len(NEW_NODES)}"

    if node_name in NEW_NODES:
        return NEW_NODES[node_name]['results']

    out_embeddings = model.embed_nodes(data_global)
    norm_existing = F.normalize(out_embeddings, dim=1)
    norm_new = F.normalize(new_embed, dim=1)

    sims = torch.matmul(norm_existing, norm_new.T).squeeze()
    top_k_safe = min(top_k, sims.shape[0])
    topk = torch.topk(sims, top_k_safe)
    top_indices = topk.indices.tolist()
    top_scores = topk.values.tolist()
    linked_nodes = [nodes[i] for i in top_indices]

    # Update graph (in memory only)
    new_embed_resized = torch.zeros((1, x_global.size(1)), device=device)
    new_embed_resized[:, :min(128, x_global.size(1))] = new_embed[:, :min(128, x_global.size(1))]
    x_global = torch.cat([x_global, new_embed_resized], dim=0)
    new_index = x_global.shape[0] - 1
    top_indices_tensor = torch.tensor(top_indices, dtype=torch.long, device=device)
    new_edges = torch.stack([
        torch.full((top_k_safe,), new_index, dtype=torch.long, device=device),
        top_indices_tensor
    ], dim=0)
    data_global.edge_index = torch.cat([data_global.edge_index, new_edges], dim=1)
    data_global.edge_type = torch.cat([data_global.edge_type, torch.zeros(top_k, dtype=torch.long, device=device)])
    data_global.x = x_global
    nodes_global = nodes_global + [node_name]

    # Save prediction
    result = [{"node": n, "score": s} for n, s in zip(linked_nodes, top_scores)]
    NEW_NODES[node_name] = {"text": text, "results": result}
    return result

# ==============================
# 💾 Export Graph to CSV for External Tools
# ==============================
# def export_graph_to_csv():
#     edge_idx_np = data_global.edge_index.cpu().numpy()
#     edge_types_np = data_global.edge_type.cpu().numpy()
#     edge_type_labels = edge_encoder.inverse_transform(edge_types_np[:len(edge_types_np)])
#     rows = []
#     for i in range(edge_idx_np.shape[1]):
#         src = nodes_global[edge_idx_np[0, i]]
#         tgt = nodes_global[edge_idx_np[1, i]]
#         rel = edge_type_labels[i] if i < len(edge_type_labels) else 'unknown'
#         rows.append({'source': src, 'target': tgt, 'edge_type': rel})
#     df = pd.DataFrame(rows)
#     df.to_csv("exported_graph.csv", index=False)
#     print("✅ Exported full graph to exported_graph.csv")

# ==============================
# 💾 Persist Graph to Disk
# ==============================
def save_graph_state():
    # export_graph_to_csv()
    torch.save({
        'bert_proj_state_dict': bert_proj.state_dict(),
        'x': x_global,
        'edge_index': data_global.edge_index,
        'edge_type': data_global.edge_type,
        'nodes': nodes_global,
        'new_nodes': NEW_NODES
    }, 'graph_state.pt')
    print("✅ Graph state saved to graph_state.pt")

# ==============================
# 🌐 REST API for Client Access
# ==============================
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data_in = request.get_json()
    text = data_in.get("text")
    node_name = data_in.get("node_name", None)
    top_k = int(data_in.get("top_k", 5))
    if not text:
        return jsonify({"error": "Missing 'text' field in request."}), 400
    results = predict_and_insert_node(text, node_name=node_name, top_k=top_k)
    save_graph_state()
    return jsonify({"new_node": node_name or f"new_node_{len(NEW_NODES)-1}", "links": results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# ==============================
# 📤 Example Standalone Prediction (optional if testing via curl/postman)
# new_text = "This work proposes a novel GNN method with attention."
# results = predict_and_insert_node(new_text, node_name="gnn_paper")
# print(json.dumps({"new_node": "gnn_paper", "links": results}, indent=2))
# save_graph_state()
