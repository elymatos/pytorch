# ==============================================
# 🔍 Inference Script for R-GAT + BERT
# ==============================================
# This script:
# - Loads a saved model and graph state
# - Predicts labels for all nodes in the graph

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
import pandas as pd
import torch.nn as nn
from torch_geometric.nn import GATConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

# ==============================
# 🔄 Projection from BERT to GNN space
bert_proj = nn.Linear(768, 128).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


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

# ==========================
# 📦 Load Saved Graph State
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔁 Loading saved graph state from disk...")
saved = torch.load('graph_state.pt')
x = saved['x']
edge_index = saved['edge_index']
edge_type = saved['edge_type']
nodes = saved['nodes']
model = saved['model']
label_encoder = saved['label_encoder']

# Load the saved checkpoint
# checkpoint = torch.load("graph_state.pt", map_location=device)
#
# # Load the model, projection layer, and graph data
# model = checkpoint["model"]
# bert_proj = checkpoint["bert_proj"]
# model.bert_proj = bert_proj
# edge_encoder = checkpoint["edge_encoder"]
# model.rgat1.rel_embed = nn.Embedding(len(edge_encoder.classes_), model.rgat1.gat.in_channels)

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

#==========================
#📊 Run Inference for All Nodes
#==========================
model.eval()
with torch.no_grad():
    # Forward pass through the model
    logits = model(data)
    preds = logits.argmax(dim=1)

    # Decode predicted labels using label_encoder
    predicted_labels = label_encoder.inverse_transform(preds.cpu().numpy())

# ==========================
# 🖨️ Show Predictions
# ==========================
results_df = pd.DataFrame({
    "node_id": nodes,
    "predicted_label": predicted_labels
})

print(results_df.head())  # Print the top 5 predictions

# Optionally, save predictions to a CSV file
results_df.to_csv("predicted_labels.csv", index=False)
print("✅ Predictions saved to predicted_labels.csv")

# ==========================
# ➕ Inference for New Node (Optional Example)
# ==========================
def predict_new_node_label(text):
    with torch.no_grad():
        # Tokenize new text and get BERT embeddings (768-dim)
        global x_global, nodes_global, data_global
        model.eval()
        inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=32)
        new_embed = bert_proj(bert(**inputs).last_hidden_state[:, 0, :].to(device))


        # inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        # bert_output = bert(**inputs).last_hidden_state[:, 0, :]  # Shape [1, 768]

        # Project BERT output (768-dim) to the same space as the graph (128-dim)
        node_feat = bert_proj(bert_output)  # Shape [1, 128] - Projected to 128-dim
        node_feat = node_feat.to(device)

        # Ensure the existing node features (data.x) are also in 128-dim (apply projection)
        existing_features = bert_proj(data.x)  # Project graph embeddings to 128-dim

        # Combine the new node's features with the existing nodes' features (both are now 128-dim)
        x_combined = torch.cat([existing_features, node_feat], dim=0)

        # Create dummy edges for the new node if needed (for isolated node)
        dummy_index = torch.tensor([[], []], dtype=torch.long, device=device)
        dummy_type = torch.tensor([], dtype=torch.long, device=device)

        # Augment data with the new node's features
        data_augmented = Data(x=x_combined, edge_index=data.edge_index, edge_type=data.edge_type)

        # Forward pass through the model
        logits = model(data_augmented)  # Forward through the GNN layers
        logits = logits[-1]  # Get the last node's logits
        probs = F.softmax(logits, dim=0)  # Apply softmax to get probabilities

        # Get the predicted class
        pred_class = torch.argmax(probs).item()
        return label_encoder.inverse_transform([pred_class])[0], probs.cpu().numpy()

# ==========================
# 📈 Run Inference for New Node
# ==========================
# new_text = "This study investigates graph neural networks for molecule property prediction."
# predicted_label, probabilities = predict_new_node_label(new_text)
# print(f"🆕 New node prediction: {predicted_label}")
# print(f"Probabilities: {probabilities}")
