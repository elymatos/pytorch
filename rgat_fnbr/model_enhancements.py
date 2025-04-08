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
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report

from framenet_rgat import FrameNetDataProcessor, RGAT, LinkPredictionTask


class AdvancedFeatureExtractor:
    """
    Extract advanced features for frames and lexical units
    using various NLP techniques and pre-trained models.
    """

    def __init__(self, use_bert=True, use_pos=True, use_graph_features=True):
        self.use_bert = use_bert
        self.use_pos = use_pos
        self.use_graph_features = use_graph_features

        if use_bert:
            # Load BERT model and tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = BertModel.from_pretrained('bert-base-multilingual-cased')

    def extract_text_embeddings(self, text_list):
        """Extract BERT embeddings for a list of texts"""
        if not self.use_bert:
            return None

        embeddings = []

        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use CLS token embedding or average of all tokens
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
                embeddings.append(batch_embeddings)

        # Concatenate all batches
        if embeddings:
            all_embeddings = torch.cat(embeddings, dim=0)
            return all_embeddings
        else:
            return None

    def extract_pos_features(self, lemmas):
        """Extract part-of-speech features from lemmas"""
        if not self.use_pos:
            return None

        # Simple POS extraction based on suffixes commonly used in FrameNet
        pos_features = []
        pos_mapping = {
            'v': [1, 0, 0, 0],  # verb
            'n': [0, 1, 0, 0],  # noun
            'a': [0, 0, 1, 0],  # adjective
            'adv': [0, 0, 0, 1],  # adverb
        }

        for lemma in lemmas:
            pos = 'n'  # default

            # Check for common POS markers in FrameNet
            if '.' in lemma:
                parts = lemma.split('.')
                if len(parts) > 1:
                    pos_marker = parts[-1].lower()
                    if pos_marker in pos_mapping:
                        pos = pos_marker

            pos_features.append(pos_mapping.get(pos, [0, 0, 0, 0]))

        return torch.tensor(pos_features, dtype=torch.float)

    def extract_graph_features(self, relation_df, node_ids):
        """Extract graph-based features like node degree, centrality, etc."""
        if not self.use_graph_features:
            return None

        # Count incoming and outgoing connections for each node
        in_degree = {node_id: 0 for node_id in node_ids}
        out_degree = {node_id: 0 for node_id in node_ids}

        for _, row in relation_df.iterrows():
            source_id = row['source_id']
            target_id = row['target_id']

            if source_id in out_degree:
                out_degree[source_id] += 1

            if target_id in in_degree:
                in_degree[target_id] += 1

        # Create feature matrix
        graph_features = []
        for node_id in node_ids:
            node_features = [
                in_degree[node_id],
                out_degree[node_id],
                in_degree[node_id] + out_degree[node_id]  # total degree
            ]
            graph_features.append(node_features)

        return torch.tensor(graph_features, dtype=torch.float)


class EnhancedFrameNetDataProcessor(FrameNetDataProcessor):
    """
    Enhanced data processor with advanced feature extraction capabilities
    """

    def __init__(self, frames_csv, lexical_units_csv, relations_csv,
                 use_bert=True, use_pos=True, use_graph_features=True):
        super().__init__(frames_csv, lexical_units_csv, relations_csv)
        self.feature_extractor = AdvancedFeatureExtractor(
            use_bert=use_bert,
            use_pos=use_pos,
            use_graph_features=use_graph_features
        )

    def extract_node_features(self, frames_df, lu_df):
        """Extract advanced features for frames and lexical units"""
        # Get all nodes
        all_nodes = []
        for idx in range(len(self.node_to_id_mapping)):
            node_id, node_name, node_type = self.node_to_id_mapping[idx]
            all_nodes.append((node_id, node_name, node_type))

        # Separate frames and LUs
        frame_indices = [idx for idx, (_, _, node_type) in enumerate(all_nodes) if node_type == 0]
        lu_indices = [idx for idx, (_, _, node_type) in enumerate(all_nodes) if node_type == 1]

        # Get text for BERT embeddings
        texts = []
        lemmas = []
        node_ids = []

        for node_id, node_name, node_type in all_nodes:
            if node_type == 0:  # Frame
                frame_row = frames_df[frames_df['frame_id'] == node_id]
                if not frame_row.empty:
                    text = frame_row.iloc[0]['frame_definition']
                    texts.append(text)
                    lemmas.append('')
                    node_ids.append(node_id)
            else:  # Lexical Unit
                lu_row = lu_df[lu_df['lu_id'] == node_id]
                if not lu_row.empty:
                    text = lu_row.iloc[0]['sense_description']
                    lemma = lu_row.iloc[0]['lemma']
                    texts.append(text)
                    lemmas.append(lemma)
                    node_ids.append(node_id)

        # Extract features
        print("Extracting text embeddings...")
        text_embeddings = self.feature_extractor.extract_text_embeddings(texts)

        print("Extracting POS features...")
        pos_features = self.feature_extractor.extract_pos_features(lemmas)

        print("Extracting graph features...")
        relations_df = pd.read_csv(self.relations_csv)
        graph_features = self.feature_extractor.extract_graph_features(relations_df, node_ids)

        # Combine features
        features = []

        # Base feature size if no advanced features are available
        base_feature_dim = 64

        for i in range(len(all_nodes)):
            node_features = []

            # Add text embeddings if available
            if text_embeddings is not None and i < text_embeddings.size(0):
                node_features.append(text_embeddings[i])

            # Add POS features if available and this is an LU
            if pos_features is not None and all_nodes[i][2] == 1:
                lu_idx = lu_indices.index(i)
                if lu_idx < pos_features.size(0):
                    node_features.append(pos_features[lu_idx])

            # Add graph features if available
            if graph_features is not None and i < graph_features.size(0):
                node_features.append(graph_features[i])

            # If no features available, use random embedding
            if not node_features:
                node_features.append(torch.randn(base_feature_dim))

            # Concatenate all features
            feature = torch.cat([f.float() for f in node_features])
            features.append(feature)

        # Get feature dimension
        feat_dim = features[0].size(0)

        # Pad or truncate all features to same dimension
        padded_features = []
        for feat in features:
            if feat.size(0) < feat_dim:
                # Pad
                padded = torch.cat([feat, torch.zeros(feat_dim - feat.size(0))])
                padded_features.append(padded)
            elif feat.size(0) > feat_dim:
                # Truncate
                padded_features.append(feat[:feat_dim])
            else:
                padded_features.append(feat)

        # Stack into tensor
        return torch.stack(padded_features)


class MultiRelationRGAT(RGAT):
    """
    Enhanced R-GAT model that handles multiple relation types more effectively
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 heads=8, dropout=0.2, relation_embedding_dim=16):
        super().__init__(in_channels, hidden_channels, out_channels, num_relations, heads, dropout)

        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations, relation_embedding_dim)

        # Additional layer for relation-specific prediction
        self.relation_predictor = nn.Sequential(
            nn.Linear(out_channels * 2 + relation_embedding_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1)
        )

    def decode_with_relation(self, z, edge_index, relation_type):
        """Decode embeddings to predict links with relation type awareness"""
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]

        # Get relation embeddings
        rel_emb = self.relation_embedding(relation_type)

        # Combine node and relation features
        z_pair = torch.cat([z_src, z_dst, rel_emb], dim=1)

        # Predict link probability
        return self.relation_predictor(z_pair)

    def forward(self, x, edge_index, edge_type, target_edges=None, target_relations=None):
        """Forward pass with relation awareness"""
        # Get node embeddings
        z = self.encode(x, edge_index, edge_type)

        # For link prediction during training
        if target_edges is not None:
            if target_relations is not None:
                # Use relation-aware prediction if relation types are provided
                return self.decode_with_relation(z, target_edges, target_relations)
            else:
                # Fall back to standard prediction
                return self.decode(z, target_edges)

        # For embedding generation during inference
        return z


class AdvancedLinkPredictionTask(LinkPredictionTask):
    """
    Advanced link prediction with relation-specific training and evaluation
    """

    def prepare_data_for_relation_prediction(self, data):
        """
        Prepare data for relation-specific link prediction
        """
        # Extract all edges and their relation types
        edge_index = data.edge_index
        edge_type = data.edge_type

        # Split edges by relation type
        relation_edges = {}
        relation_types = torch.unique(edge_type).tolist()

        for rel_type in relation_types:
            rel_mask = edge_type == rel_type
            rel_edges = edge_index[:, rel_mask]
            relation_edges[rel_type] = rel_edges

        # Split into train/val/test for each relation type
        train_data = {}
        val_data = {}
        test_data = {}

        for rel_type, rel_edges in relation_edges.items():
            num_edges = rel_edges.size(1)
            if num_edges < 3:  # Skip relations with too few edges
                continue

            perm = torch.randperm(num_edges)

            # 70% train, 15% val, 15% test
            train_idx = perm[:int(0.7 * num_edges)]
            val_idx = perm[int(0.7 * num_edges):int(0.85 * num_edges)]
            test_idx = perm[int(0.85 * num_edges):]

            train_edge_index = rel_edges[:, train_idx]
            val_edge_index = rel_edges[:, val_idx]
            test_edge_index = rel_edges[:, test_idx]

            # Create relation type tensors
            train_edge_type = torch.full((train_idx.size(0),), rel_type, dtype=torch.long)
            val_edge_type = torch.full((val_idx.size(0),), rel_type, dtype=torch.long)
            test_edge_type = torch.full((test_idx.size(0),), rel_type, dtype=torch.long)

            # Create negative samples
            # For relation-specific negative sampling, we keep the source and relation type
            # but randomly select a target node
            def create_negative_edges_with_relation(pos_edges, num_nodes, relation_type):
                neg_edges = pos_edges.clone()
                neg_relation = torch.full((pos_edges.size(1),), relation_type, dtype=torch.long)

                # Replace target nodes with random nodes
                for i in range(pos_edges.size(1)):
                    neg_edges[1, i] = torch.randint(0, num_nodes, (1,))

                return neg_edges, neg_relation

            train_neg_edge_index, train_neg_edge_type = create_negative_edges_with_relation(
                train_edge_index, data.x.size(0), rel_type)
            val_neg_edge_index, val_neg_edge_type = create_negative_edges_with_relation(
                val_edge_index, data.x.size(0), rel_type)
            test_neg_edge_index, test_neg_edge_type = create_negative_edges_with_relation(
                test_edge_index, data.x.size(0), rel_type)

            # Store data for this relation type
            train_data[rel_type] = {
                'pos_edge': train_edge_index,
                'pos_rel': train_edge_type,
                'neg_edge': train_neg_edge_index,
                'neg_rel': train_neg_edge_type
            }

            val_data[rel_type] = {
                'pos_edge': val_edge_index,
                'pos_rel': val_edge_type,
                'neg_edge': val_neg_edge_index,
                'neg_rel': val_neg_edge_type
            }

            test_data[rel_type] = {
                'pos_edge': test_edge_index,
                'pos_rel': test_edge_type,
                'neg_edge': test_neg_edge_index,
                'neg_rel': test_neg_edge_type
            }

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    def train_with_relations(self, data, relation_data, epochs=200):
        """Train the model with relation awareness"""
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_type = data.edge_type.to(self.device)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Accumulate loss across all relation types
            total_loss = 0

            # Train on each relation type
            for rel_type, rel_data in relation_data['train'].items():
                pos_edge = rel_data['pos_edge'].to(self.device)
                pos_rel = rel_data['pos_rel'].to(self.device)
                neg_edge = rel_data['neg_edge'].to(self.device)
                neg_rel = rel_data['neg_rel'].to(self.device)

                # Create positive and negative labels
                pos_label = torch.ones(pos_edge.size(1), 1).to(self.device)
                neg_label = torch.zeros(neg_edge.size(1), 1).to(self.device)

                # Combine positive and negative examples
                train_edges = torch.cat([pos_edge, neg_edge], dim=1)
                train_rels = torch.cat([pos_rel, neg_rel], dim=0)
                train_labels = torch.cat([pos_label, neg_label], dim=0)

                # Forward pass
                logits = self.model(x, edge_index, edge_type, train_edges, train_rels)
                loss = self.criterion(logits, train_labels)

                # Accumulate loss
                total_loss += loss

            # Backward pass on accumulated loss
            total_loss.backward()
            self.optimizer.step()

            # Validation
            if epoch % 10 == 0:
                val_auc = self.evaluate_with_relations(data, relation_data, 'val')
                print(f'Epoch: {epoch}, Loss: {total_loss.item():.4f}, Val AUC: {val_auc:.4f}')

    def evaluate_with_relations(self, data, relation_data, split='val'):
        """Evaluate the model with relation awareness"""
        self.model.eval()

        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_type = data.edge_type.to(self.device)

        all_labels = []
        all_preds = []

        # Evaluate on each relation type
        for rel_type, rel_data in relation_data[split].items():
            pos_edge = rel_data['pos_edge'].to(self.device)
            pos_rel = rel_data['pos_rel'].to(self.device)
            neg_edge = rel_data['neg_edge'].to(self.device)
            neg_rel = rel_data['neg_rel'].to(self.device)

            # Create labels
            pos_label = torch.ones(pos_edge.size(1))
            neg_label = torch.zeros(neg_edge.size(1))
            labels = torch.cat([pos_label, neg_label]).to(self.device)

            # Combine edges and relations
            eval_edges = torch.cat([pos_edge, neg_edge], dim=1)
            eval_rels = torch.cat([pos_rel, neg_rel], dim=0)

            with torch.no_grad():
                # Get predictions
                logits = self.model(x, edge_index, edge_type, eval_edges, eval_rels)
                pred = torch.sigmoid(logits).squeeze()

            # Accumulate predictions and labels
            all_labels.append(labels.cpu())
            all_preds.append(pred.cpu())

        # Combine all predictions
        if all_labels and all_preds:
            all_labels = torch.cat(all_labels).numpy()
            all_preds = torch.cat(all_preds).numpy()

            # Calculate AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(all_labels, all_preds)
                return auc
            except:
                return 0.5  # Default value if calculation fails
        else:
            return 0.5

    def predict_lu_frame_links_with_relations(self, data, lu_indices, relation_type, top_k=5):
        """
        Predict potential frame links for given lexical units with a specific relation type
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
            num_frames = len(frame_indices)
            candidate_edges = torch.zeros((2, num_frames), dtype=torch.long, device=self.device)
            candidate_edges[0] = lu_idx  # Source node (LU)
            candidate_edges[1] = frame_indices  # Target nodes (frames)

            # Create relation type tensor
            candidate_relations = torch.full((num_frames,), relation_type, dtype=torch.long, device=self.device)

            # Predict scores
            with torch.no_grad():
                scores = self.model(x, edge_index, edge_type, candidate_edges, candidate_relations)
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


def evaluate_model(model, data, link_data, processor, output_dir="evaluation_results"):
    """
    Comprehensive evaluation of the model with various metrics and visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Task handler
    task = AdvancedLinkPredictionTask(model)

    # Test AUC
    test_auc = task.evaluate(data, link_data, 'test')
    print(f'Test AUC: {test_auc:.4f}')

    # Get predictions and labels for test set
    model.eval()

    # Move data to device
    device = task.device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    test_pos = link_data['test_pos'].to(device)
    test_neg = link_data['test_neg'].to(device)

    pos_label = torch.ones(test_pos.size(1))
    neg_label = torch.zeros(test_neg.size(1))
    labels = torch.cat([pos_label, neg_label]).numpy()

    test_edges = torch.cat([test_pos, test_neg], dim=1)

    with torch.no_grad():
        logits = model(x, edge_index, edge_type, test_edges)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Set threshold
    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    # Classification report
    report = classification_report(labels, preds)
    print("Classification Report:")
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Link', 'Link'],
                yticklabels=['No Link', 'Link'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()

    # Visualize some example predictions for new lexical units
    # Select a few random LUs for visualization
    lu_indices = torch.where(data.node_type == 1)[0][:5]  # First 5 LUs

    # Predict frame links
    predictions = task.predict_lu_frame_links(data, lu_indices)

    # Create visualization
    plt.figure(figsize=(12, 8))

    for i, lu_idx in enumerate(lu_indices):
        lu_id, lu_name, _ = processor.node_to_id_mapping[lu_idx.item()]

        # Get predictions
        frame_preds = predictions[lu_idx.item()]

        # Plot
        frame_names = [processor.node_to_id_mapping[frame_idx][1] for frame_idx, _ in frame_preds]
        scores = [score for _, score in frame_preds]

        plt.subplot(len(lu_indices), 1, i + 1)
        plt.barh(frame_names, scores)
        plt.title(f'LU: {lu_name}')
        plt.xlim([0, 1])
        plt.tight_layout()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle('Frame Predictions for Example Lexical Units')
    plt.savefig(os.path.join(output_dir, "example_predictions.png"))
    plt.close()

    return {
        'test_auc': test_auc,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
    }


def main_enhanced():
    """
    Enhanced main function with advanced features
    """
    # File paths (adjust these to your actual file paths)
    frames_csv = "frames.csv"
    lexical_units_csv = "lexical_units.csv"
    relations_csv = "relations.csv"

    # Process data with enhanced features
    processor = EnhancedFrameNetDataProcessor(
        frames_csv,
        lexical_units_csv,
        relations_csv,
        use_bert=True,  # Set to False if BERT is not available
        use_pos=True,
        use_graph_features=True
    )
    data, num_relations = processor.process()

    # Create enhanced model
    in_channels = data.x.size(1)
    hidden_channels = 128
    out_channels = 64
    model = MultiRelationRGAT(
        in_channels,
        hidden_channels,
        out_channels,
        num_relations,
        heads=8,
        dropout=0.3,
        relation_embedding_dim=32
    )

    # Set up link prediction task
    task = AdvancedLinkPredictionTask(model)

    # Option 1: Standard link prediction
    link_data = task.prepare_data_for_link_prediction(data)
    task.train(data, link_data, epochs=200)

    # Option 2: Relation-specific link prediction
    relation_data = task.prepare_data_for_relation_prediction(data)
    task.train_with_relations(data, relation_data, epochs=200)

    # Evaluate model
    metrics = evaluate_model(model, data, link_data, processor)

    # Example: Predict inheritance links for new lexical units
    # Find the index of the 'Inheritance' relation type
    inheritance_type = 0  # This should be set to the actual index of the inheritance relation

    # Get new LU indices (for demonstration, use existing LUs)
    new_lu_indices = torch.where(data.node_type == 1)[0][:10]  # First 10 LUs

    # Predict frame links with relation awareness
    predictions = task.predict_lu_frame_links_with_relations(
        data, new_lu_indices, inheritance_type, top_k=5)

    # Print predictions
    for lu_idx, frame_preds in predictions.items():
        lu_id, lu_name, _ = processor.node_to_id_mapping[lu_idx]
        print(f"\nTop frame predictions for LU '{lu_name}' (ID: {lu_id}):")
        for frame_idx, score in frame_preds:
            frame_id, frame_name, _ = processor.node_to_id_mapping[frame_idx]
            print(f"  - {frame_name} (ID: {frame_id}): {score:.4f}")

    return metrics


def visualize_graph(data, processor, output_dir="graph_visualizations"):
    """
    Create visualizations of the FrameNet graph structure
    """
    import networkx as nx
    from matplotlib.cm import get_cmap

    os.makedirs(output_dir, exist_ok=True)

    # Convert PyG data to NetworkX
    G = nx.DiGraph()

    # Add nodes
    for idx in range(len(processor.node_to_id_mapping)):
        node_id, node_name, node_type = processor.node_to_id_mapping[idx]
        node_type_name = "Frame" if node_type == 0 else "LexicalUnit"
        G.add_node(idx, id=node_id, name=node_name, type=node_type_name)

    # Add edges
    edge_index = data.edge_index.cpu().numpy()
    edge_types = data.edge_type.cpu().numpy()

    # Map relation type indices to names (this would need actual relation names)
    relation_names = {i: f"Relation_{i}" for i in range(max(edge_types) + 1)}
    # If you have actual relation names:
    # relation_names = {i: name for i, name in enumerate(processor.relation_encoder.classes_)}

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        rel_type = edge_types[i]
        G.add_edge(src, dst, type=relation_names[rel_type])

    # 1. Overall graph visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    # Node colors by type
    node_colors = ["skyblue" if G.nodes[n]["type"] == "Frame" else "lightgreen" for n in G]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8)

    # Draw edges with colors by relation type
    cmap = get_cmap("tab10")
    unique_relations = set(nx.get_edge_attributes(G, "type").values())
    relation_colors = {rel: cmap(i / len(unique_relations)) for i, rel in enumerate(unique_relations)}

    for relation in unique_relations:
        # Get edges of this relation type
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == relation]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=0.5,
                               edge_color=[relation_colors[relation]] * len(edges),
                               alpha=0.6, arrows=True, arrowsize=10)

    # Add legend for relation types
    legend_handles = [plt.Line2D([0], [0], color=relation_colors[rel], lw=2, label=rel)
                      for rel in unique_relations]
    node_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
                               markersize=10, label='Frame'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                               markersize=10, label='Lexical Unit')]

    plt.legend(handles=legend_handles + node_handles, loc='upper right')
    plt.title("FrameNet Semantic Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complete_graph.png"), dpi=300)
    plt.close()

    # 2. Subgraph for better visualization - select only a portion of the graph
    # Find a frame with several LUs
    frame_counts = {}
    for u, v, d in G.edges(data=True):
        if d["type"] == "Inheritance":  # Assuming "Inheritance" is the LU->Frame relation
            frame = v
            if frame not in frame_counts:
                frame_counts[frame] = 0
            frame_counts[frame] += 1

    # Sort frames by number of LUs
    top_frames = sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    if top_frames:
        # Create a subgraph with the top frames and their LUs
        subgraph_nodes = set()
        for frame, _ in top_frames:
            subgraph_nodes.add(frame)
            # Add connected LUs
            for u, v in G.edges():
                if v == frame and G.edges[u, v]["type"] == "Inheritance":
                    subgraph_nodes.add(u)

        subgraph = G.subgraph(subgraph_nodes)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph, seed=42)

        # Node colors and sizes
        node_colors = ["skyblue" if subgraph.nodes[n]["type"] == "Frame" else "lightgreen" for n in subgraph]
        node_sizes = [100 if subgraph.nodes[n]["type"] == "Frame" else 50 for n in subgraph]

        # Draw nodes with labels
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

        # Draw edges
        for relation in unique_relations:
            edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d["type"] == relation]
            if edges:
                nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=1.0,
                                       edge_color=[relation_colors[relation]] * len(edges),
                                       alpha=0.7, arrows=True, arrowsize=15)

        # Add labels for frames only to avoid cluttering
        frame_labels = {n: subgraph.nodes[n]["name"] for n in subgraph
                        if subgraph.nodes[n]["type"] == "Frame"}
        nx.draw_networkx_labels(subgraph, pos, labels=frame_labels, font_size=10)

        # Add legend
        legend_handles = [plt.Line2D([0], [0], color=relation_colors[rel], lw=2, label=rel)
                          for rel in unique_relations if any(d["type"] == rel for _, _, d in subgraph.edges(data=True))]
        node_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
                                   markersize=10, label='Frame'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                                   markersize=10, label='Lexical Unit')]

        plt.legend(handles=legend_handles + node_handles, loc='upper right')
        plt.title("FrameNet Subgraph: Top Frames with Connected Lexical Units")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "subgraph_visualization.png"), dpi=300)
        plt.close()

    # 3. Create embedding visualization using t-SNE
    from sklearn.manifold import TSNE

    # Get node embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiRelationRGAT(data.x.size(1), 128, 64, len(relation_names))
    model.to(device)

    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(x, edge_index, edge_type).cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    node_tsne = tsne.fit_transform(embeddings)

    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))

    # Separate frames and LUs
    frame_mask = np.array([G.nodes[n]["type"] == "Frame" for n in G.nodes()])
    lu_mask = ~frame_mask

    plt.scatter(node_tsne[frame_mask, 0], node_tsne[frame_mask, 1],
                c='skyblue', label='Frames', alpha=0.7, s=80)
    plt.scatter(node_tsne[lu_mask, 0], node_tsne[lu_mask, 1],
                c='lightgreen', label='Lexical Units', alpha=0.5, s=40)

    # Add some labels for larger frames
    for i, node_idx in enumerate(G.nodes()):
        if G.nodes[node_idx]["type"] == "Frame" and frame_counts.get(node_idx, 0) > 3:
            plt.annotate(G.nodes[node_idx]["name"],
                         (node_tsne[i, 0], node_tsne[i, 1]),
                         fontsize=8)

    plt.title('t-SNE Visualization of Node Embeddings')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "tsne_embeddings.png"), dpi=300)
    plt.close()

    return G


if __name__ == "__main__":
    # Run enhanced version
    main_enhanced()