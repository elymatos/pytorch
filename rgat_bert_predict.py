# ==========================================
# 🔗 Predict Links for New Node (Static Graph)
# ==========================================
# Goal: Load saved R-GAT model, insert new node with predefined edges,
# and predict new potential links WITHOUT updating the graph.

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
import pandas as pd
import json

# ==============================
# ⚙️ Load Resources
# ==============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved graph state
graph = torch.load('graph_state.pt', map_location=device)
x = graph['x'].to(device)
edge_index = graph['edge_index'].to(device)
edge_type = graph['edge_type'].to(device)
nodes = graph['nodes']
NEW_NODES = graph.get('new_nodes', {})

# Load model
from rgat_model_def import RGATModel  # assume you saved the model class

model = RGATModel(in_dim=x.size(1), hidden_dim=128, out_dim=3, num_relations=edge_type.max().item() + 1).to(device)
checkpoint = torch.load('rgat_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load BERT & projection layer
bert_proj = torch.nn.Linear(768, 128).to(device)
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_proj.load_state_dict(checkpoint['bert_proj_state_dict'])
bert.eval()

# ==============================
# 📥 New Node Inputs
# ==============================
new_node_text = "A study on transformers for citation analysis."
predefined_links = ["doc1", "doc3"]  # known edges from new node to existing nodes

# Map existing nodes to indices
node_map = {node: idx for idx, node in enumerate(nodes)}
linked_indices = [node_map[n] for n in predefined_links if n in node_map]

# ==============================
# 🔍 Encode New Node with BERT
# ==============================
with torch.no_grad():
    inputs = tokenizer([new_node_text], return_tensors='pt', padding=True, truncation=True, max_length=32)
    bert_out = bert(**inputs.to(device)).last_hidden_state[:, 0, :]
    new_embed = bert_proj(bert_out)  # shape: [1, 128]

# ==============================
# 🧠 Get Node Embeddings (from model)
# ==============================
from torch_geometric.data import Data

data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
with torch.no_grad():
    node_embeddings = model.embed_nodes(data)
    norm_existing = F.normalize(node_embeddings, dim=1)
    norm_new = F.normalize(new_embed, dim=1)

# ==============================
# 🔗 Mask known connections
# ==============================
all_indices = torch.arange(len(nodes), device=device)
mask = torch.ones_like(all_indices, dtype=torch.bool)
mask[linked_indices] = False

# ==============================
# 🔮 Predict New Links
# ==============================
scores = torch.matmul(norm_existing, norm_new.T).squeeze()
scores = scores[mask]
candidates = all_indices[mask]
top_k = min(5, scores.size(0))
top_scores, top_ids = torch.topk(scores, top_k)
top_indices = candidates[top_ids]

# ==============================
# 📤 Output Results
# ==============================
results = [{
    "node": nodes[i],
    "score": round(s.item(), 4)
} for i, s in zip(top_indices.tolist(), top_scores.tolist())]

print(json.dumps({
    "new_node_text": new_node_text,
    "known_links": predefined_links,
    "predicted_new_links": results
}, indent=2))
