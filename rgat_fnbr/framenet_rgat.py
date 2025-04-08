import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class FrameNetDataProcessor:
    """
    Process FrameNet Brasil data from CSV files to create a graph representation
    suitable for R-GAT model training.
    """

    def __init__(self, frames_csv, lexical_units_csv, relations_csv):
        self.frames_csv = frames_csv
        self.lexical_units_csv = lexical_units_csv
        self.relations_csv = relations_csv

        # Encoders for categorical data
        self.node_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        # Node feature matrices
        self.node_features = None
        self.edge_index = None
        self.edge_type = None

        # Node type indicators (0 for frames, 1 for lexical units)
        self.node_types = None

        # Mappings
        self.id_to_node_mapping = {}
        self.node_to_id_mapping = {}

    def load_data(self):
        """Load data from CSV files"""
        # Load frames data
        print(f"Loading frames from {self.frames_csv}")
        frames_df = pd.read_csv(self.frames_csv)

        # Load lexical units data
        print(f"Loading lexical units from {self.lexical_units_csv}")
        lu_df = pd.read_csv(self.lexical_units_csv)

        # Load relations data
        print(f"Loading relations from {self.relations_csv}")
        relations_df = pd.read_csv(self.relations_csv)

        return frames_df, lu_df, relations_df

    def create_node_mappings(self, frames_df, lu_df):
        """Create mappings between node IDs and their indices in the graph"""
        # Create a unique identifier for each node
        frames_with_type = [(row['frame_id'], row['frame_name'], 0) for _, row in frames_df.iterrows()]
        lus_with_type = [(row['lu_id'], row['lu_name'], 1) for _, row in lu_df.iterrows()]

        # Combine all nodes
        all_nodes = frames_with_type + lus_with_type

        # Create mappings
        for idx, (node_id, node_name, node_type) in enumerate(all_nodes):
            self.id_to_node_mapping[node_id] = idx
            self.node_to_id_mapping[idx] = (node_id, node_name, node_type)

        # Create node type indicators
        self.node_types = torch.tensor([node_type for _, _, node_type in all_nodes], dtype=torch.long)

        return len(all_nodes)

    def extract_node_features(self, frames_df, lu_df):
        """Extract features for each node in the graph"""
        # For demonstration purposes, we'll use simple one-hot encoding for now
        # In a real implementation, you might want to use text embeddings from frame definitions
        # and lexical unit sense descriptions

        num_nodes = len(self.id_to_node_mapping)
        feature_dim = 64  # Can be adjusted based on your needs

        # Initialize with random embeddings
        # In a real implementation, you'd replace this with actual embeddings
        node_features = torch.randn(num_nodes, feature_dim)

        return node_features

    def create_edge_lists(self, relations_df):
        """Create edge indices and types for the graph"""
        edge_list = []
        edge_types = []

        # Encode relation types
        unique_relations = relations_df['relation_type'].unique()
        self.relation_encoder.fit(unique_relations)

        # Create edges from relations
        for _, row in relations_df.iterrows():
            source_id = row['source_id']
            target_id = row['target_id']
            relation_type = row['relation_type']

            # Convert IDs to indices
            if source_id in self.id_to_node_mapping and target_id in self.id_to_node_mapping:
                source_idx = self.id_to_node_mapping[source_id]
                target_idx = self.id_to_node_mapping[target_id]

                # Add edge
                edge_list.append([source_idx, target_idx])

                # Add edge type
                edge_type = self.relation_encoder.transform([relation_type])[0]
                edge_types.append(edge_type)

        # Create tensor for edge index (PyTorch Geometric format)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_type, len(unique_relations)

    def process(self):
        """Process all data and create graph representation"""
        # Load data
        frames_df, lu_df, relations_df = self.load_data()

        # Create node mappings
        num_nodes = self.create_node_mappings(frames_df, lu_df)
        print(f"Created mappings for {num_nodes} nodes ({len(frames_df)} frames, {len(lu_df)} lexical units)")

        # Extract node features
        self.node_features = self.extract_node_features(frames_df, lu_df)
        print(f"Created node features with shape {self.node_features.shape}")

        # Create edge lists
        self.edge_index, self.edge_type, num_relations = self.create_edge_lists(relations_df)
        print(f"Created {self.edge_index.size(1)} edges with {num_relations} relation types")

        # Create PyTorch Geometric Data object
        data = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            node_type=self.node_types,
            num_relations=torch.tensor(num_relations, dtype=torch.long)
        )

        return data, num_relations


class RGAT(nn.Module):
    """
    Relational Graph Attention Network for FrameNet link prediction
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, heads=8, dropout=0.2):
        super(RGAT, self).__init__()

        self.conv1 = RGATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            heads=heads,
            dropout=dropout
        )

        self.conv2 = RGATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            num_relations=num_relations,
            heads=1,
            dropout=dropout
        )

        # Link prediction layer
        self.link_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1)
        )

    def encode(self, x, edge_index, edge_type):
        """Encode nodes using R-GAT layers"""
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, z, edge_index):
        """Decode embeddings to predict links"""
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        z_pair = torch.cat([z_src, z_dst], dim=1)

        # Predict link probability
        return self.link_predictor(z_pair)

    def forward(self, x, edge_index, edge_type, target_edges=None):
        """Forward pass"""
        # Get node embeddings
        z = self.encode(x, edge_index, edge_type)

        # For link prediction during training
        if target_edges is not None:
            return self.decode(z, target_edges)

        # For embedding generation during inference
        return z


class LinkPredictionTask:
    """
    Handle the training and evaluation of the R-GAT model for link prediction
    """

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion = nn.BCEWithLogitsLoss()

    def prepare_data_for_link_prediction(self, data):
        """
        Prepare data for link prediction by splitting existing links into train/test sets
        and generating negative samples
        """
        # Extract inheritance relations (we're focusing on LU-to-Frame inheritance)
        edge_index = data.edge_index
        edge_type = data.edge_type

        # Get inheritance relation type index (assuming it's in the data)
        # You'll need to modify this based on your actual relation encodings
        inheritance_type = 0  # Change this to your actual inheritance relation index

        # Filter edges to only include inheritance relations
        inheritance_mask = edge_type == inheritance_type
        inheritance_edges = edge_index[:, inheritance_mask]

        # Split into train/val/test
        num_edges = inheritance_edges.size(1)
        perm = torch.randperm(num_edges)

        # 70% train, 15% val, 15% test
        train_idx = perm[:int(0.7 * num_edges)]
        val_idx = perm[int(0.7 * num_edges):int(0.85 * num_edges)]
        test_idx = perm[int(0.85 * num_edges):]

        train_edge_index = inheritance_edges[:, train_idx]
        val_edge_index = inheritance_edges[:, val_idx]
        test_edge_index = inheritance_edges[:, test_idx]

        # Create negative samples for training
        # For each positive edge, we create a negative edge by keeping the source
        # node (lexical unit) and replacing the target node with a random frame
        def create_negative_edges(pos_edges, num_nodes):
            neg_edges = pos_edges.clone()
            # Assume source nodes are lexical units and target nodes are frames
            # Replace target nodes with random frames
            # This is a simplified approach and might need refinement
            for i in range(pos_edges.size(1)):
                neg_edges[1, i] = torch.randint(0, num_nodes, (1,))
            return neg_edges

        train_neg_edge_index = create_negative_edges(train_edge_index, data.x.size(0))
        val_neg_edge_index = create_negative_edges(val_edge_index, data.x.size(0))
        test_neg_edge_index = create_negative_edges(test_edge_index, data.x.size(0))

        return {
            'train_pos': train_edge_index,
            'train_neg': train_neg_edge_index,
            'val_pos': val_edge_index,
            'val_neg': val_neg_edge_index,
            'test_pos': test_edge_index,
            'test_neg': test_neg_edge_index
        }

    def train(self, data, link_data, epochs=200):
        """Train the model"""
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_type = data.edge_type.to(self.device)

        train_pos = link_data['train_pos'].to(self.device)
        train_neg = link_data['train_neg'].to(self.device)

        # Create positive and negative labels
        pos_label = torch.ones(train_pos.size(1), 1).to(self.device)
        neg_label = torch.zeros(train_neg.size(1), 1).to(self.device)

        # Combine positive and negative examples
        train_edges = torch.cat([train_pos, train_neg], dim=1)
        train_labels = torch.cat([pos_label, neg_label], dim=0)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(x, edge_index, edge_type, train_edges)
            loss = self.criterion(logits, train_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Validation
            if epoch % 10 == 0:
                val_auc = self.evaluate(data, link_data, 'val')
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}')

    def evaluate(self, data, link_data, split='val'):
        """Evaluate the model"""
        self.model.eval()

        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_type = data.edge_type.to(self.device)

        # Get positive and negative edges for the specified split
        pos_edge = link_data[f'{split}_pos'].to(self.device)
        neg_edge = link_data[f'{split}_neg'].to(self.device)

        # Create labels
        pos_label = torch.ones(pos_edge.size(1))
        neg_label = torch.zeros(neg_edge.size(1))
        labels = torch.cat([pos_label, neg_label]).to(self.device)

        # Combine edges
        eval_edges = torch.cat([pos_edge, neg_edge], dim=1)

        with torch.no_grad():
            # Get predictions
            logits = self.model(x, edge_index, edge_type, eval_edges)
            pred = torch.sigmoid(logits).cpu().numpy()

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.cpu().numpy(), pred)

        return auc

    def predict_lu_frame_links(self, data, lu_indices, top_k=5):
        """
        Predict potential frame links for given lexical units

        Args:
            data: PyTorch Geometric Data object
            lu_indices: Indices of lexical units to predict frames for
            top_k: Number of top predictions to return

        Returns:
            Dictionary mapping LU indices to lists of (frame_idx, score) tuples
        """
        self.model.eval()

        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_type = data.edge_type.to(self.device)

        # Get node embeddings
        with torch.no_grad():
            node_embeddings = self.model.encode(x, edge_index, edge_type)

        # Find frame indices (node type 0)
        frame_indices = torch.where(data.node_type == 0)[0]

        predictions = {}

        for lu_idx in lu_indices:
            # Create candidate edges between this LU and all frames
            candidate_edges = torch.zeros((2, len(frame_indices)), dtype=torch.long, device=self.device)
            candidate_edges[0] = lu_idx  # Source node (LU)
            candidate_edges[1] = frame_indices  # Target nodes (frames)

            # Predict scores
            with torch.no_grad():
                scores = self.model.decode(node_embeddings, candidate_edges)
                scores = torch.sigmoid(scores).squeeze()

            # Get top-k predictions
            if len(scores.shape) == 0:  # Handle case with only one frame
                top_scores = scores.unsqueeze(0)
                top_indices = torch.tensor([0], device=self.device)
            else:
                top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

            # Store predictions
            top_frames = frame_indices[top_indices.cpu()].tolist()
            top_scores = top_scores.cpu().tolist()

            predictions[lu_idx.item()] = list(zip(top_frames, top_scores))

        return predictions


def main():
    # File paths (adjust these to your actual file paths)
    frames_csv = "frames.csv"
    lexical_units_csv = "lexical_units.csv"
    relations_csv = "relations.csv"

    # Process data
    processor = FrameNetDataProcessor(frames_csv, lexical_units_csv, relations_csv)
    data, num_relations = processor.process()

    # Create model
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 32
    model = RGAT(in_channels, hidden_channels, out_channels, num_relations)

    # Set up link prediction task
    task = LinkPredictionTask(model)

    # Prepare data for link prediction
    link_data = task.prepare_data_for_link_prediction(data)

    # Train model
    task.train(data, link_data, epochs=200)

    # Evaluate model
    test_auc = task.evaluate(data, link_data, 'test')
    print(f'Test AUC: {test_auc:.4f}')

    # Example: Predict frame links for new lexical units
    # In a real scenario, you would have indices of new LUs
    new_lu_indices = torch.tensor([10, 20, 30])  # Example indices
    predictions = task.predict_lu_frame_links(data, new_lu_indices)

    # Print predictions
    for lu_idx, frame_preds in predictions.items():
        lu_name = processor.node_to_id_mapping[lu_idx][1]
        print(f"\nTop frame predictions for LU '{lu_name}':")
        for frame_idx, score in frame_preds:
            frame_name = processor.node_to_id_mapping[frame_idx][1]
            print(f"  - {frame_name}: {score:.4f}")


if __name__ == "__main__":
    main()