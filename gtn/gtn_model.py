import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_scatter import scatter_mean, scatter_add


class LearneablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for nodes in the graph.
    """

    def __init__(self, dim, max_nodes=100):
        super(LearneablePositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_nodes, dim)

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Get position index for each node within its graph
        positions = torch.zeros_like(batch)
        for i in torch.unique(batch):
            mask = batch == i
            positions[mask] = torch.arange(mask.sum(), device=x.device)

        # Get position embeddings
        pos_embeddings = self.embedding(positions)

        # Add position embeddings to node features
        return x + pos_embeddings


class GraphTransformerLayer(nn.Module):
    """
    A Graph Transformer layer with structure-aware attention.
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, edge_dim=None, use_part_whole=True):
        super(GraphTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.use_part_whole = use_part_whole

        # Transformer convolution
        # self.transformer_conv = TransformerConv(
        #     in_dim,
        #     out_dim // heads,
        #     heads=heads,
        #     dropout=dropout,
        #     edge_dim=edge_dim,
        #     beta=True  # Enable edge bias
        # )
        # inside GraphTransformerLayer.__init__:
        # note return_attention_weights=True
        self.transformer_conv = TransformerConv(
            in_dim,
            out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,
            return_attention_weights=True
        )

        # Part-whole specific transformation
        if use_part_whole:
            self.part_whole_linear = nn.Linear(in_dim, out_dim)
            self.part_whole_attention = nn.Parameter(torch.Tensor(1, heads, out_dim // heads))
            nn.init.xavier_uniform_(self.part_whole_attention)

        # Layer normalization and skip connection components
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr=None, part_of_mask=None, batch=None):
        """
        Forward pass through the Graph Transformer layer.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            part_of_mask: Boolean mask indicating part-of relationships [num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Apply transformer convolution
        # x_trans = self.transformer_conv(x, edge_index, edge_attr)
        # Apply transformer convolution _and_ grab the per-edge attention weights
        (x_trans, (att_edge_index, att_weights)) = \
             self.transformer_conv(x, edge_index, edge_attr, return_attention_weights=True)
        # att_edge_index is [2, E], att_weights is [E, heads

        # Process part-whole relationships specially if enabled
        if self.use_part_whole and part_of_mask is not None and part_of_mask.sum() > 0:
            # Get source nodes (parts) and target nodes (wholes) for part-of edges
            part_edges = edge_index[:, part_of_mask]
            parts = part_edges[0]  # Source nodes (parts)
            wholes = part_edges[1]  # Target nodes (wholes)

            # Transform part features
            part_features = self.part_whole_linear(x[parts])

            # Apply attention to part features
            part_features = part_features.view(-1, self.heads, self.out_dim // self.heads)
            attention_scores = torch.sum(part_features * self.part_whole_attention, dim=-1)
            attention_weights = F.softmax(attention_scores, dim=0)

            # Apply attention weights
            attended_parts = part_features * attention_weights.unsqueeze(-1)
            attended_parts = attended_parts.view(-1, self.out_dim)

            # Scatter the attended part features to their corresponding wholes
            # Initialize with zeros
            part_contribution = torch.zeros_like(x_trans)

            # Add contribution from each part to its whole
            scatter_idx = wholes

            # Use scatter_add to sum contributions from parts to wholes
            part_contribution.scatter_add_(0, scatter_idx.unsqueeze(-1).expand(-1, self.out_dim), attended_parts)

            # Combine with transformer output
            x_trans = x_trans + part_contribution

        # First residual connection and normalization
        x = self.layer_norm1(x + x_trans)

        # Feed-forward network
        x_ffn = self.ffn(x)

        # Second residual connection and normalization
        x = self.layer_norm2(x + x_ffn)

        return x


class StructuralPositionalEncoding(nn.Module):
    """
    Structural positional encoding based on graph structure.
    """

    def __init__(self, dim, max_nodes=100, use_centrality=True, use_part_whole=True):
        super(StructuralPositionalEncoding, self).__init__()
        self.dim = dim
        self.use_centrality = use_centrality
        self.use_part_whole = use_part_whole

        # Learnable embeddings for different structural roles
        self.learnable_pe = LearneablePositionalEncoding(dim, max_nodes)

        # Centrality encoding
        if use_centrality:
            self.centrality_embedding = nn.Linear(1, dim)

        # Part-whole hierarchy encoding
        if use_part_whole:
            self.hierarchy_embedding = nn.Embedding(10, dim)  # Assume max hierarchy depth of 10

        # Combine different encodings
        self.combine = nn.Linear(dim * (1 + use_centrality + use_part_whole), dim)

    def forward(self, x, edge_index, part_of_mask=None, batch=None):
        """
        Compute structural positional encodings.

        Args:
            x: Node features [num_nodes, dim]
            edge_index: Graph connectivity [2, num_edges]
            part_of_mask: Boolean mask indicating part-of relationships [num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Positional encodings [num_nodes, dim]
        """
        num_nodes = x.size(0)
        encodings = []

        # Basic positional encoding
        basic_pe = self.learnable_pe(x, batch)
        encodings.append(basic_pe)

        # Centrality-based encoding (degree centrality)
        if self.use_centrality:
            # Calculate degree for each node
            edge_index_flattened = edge_index.view(-1)
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, edge_index_flattened, torch.ones_like(edge_index_flattened, dtype=torch.float))
            degree = degree.unsqueeze(-1) / (degree.max() + 1e-8)  # Normalize

            # Convert to embedding
            centrality_pe = self.centrality_embedding(degree)
            encodings.append(centrality_pe)

        # Part-whole hierarchy encoding
        if self.use_part_whole and part_of_mask is not None:
            # Calculate hierarchy level for each node
            hierarchy_level = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

            # Extract part-of edges
            if part_of_mask.sum() > 0:
                part_edges = edge_index[:, part_of_mask]

                # Iteratively propagate hierarchy levels (BFS-like approach)
                for i in range(9):  # Max depth - 1
                    # Find parts at current level
                    current_parts = (hierarchy_level == i).nonzero().squeeze(-1)

                    # Find their wholes
                    mask = torch.isin(part_edges[0], current_parts)
                    wholes = part_edges[1, mask]

                    # Update hierarchy level of wholes (if not already set)
                    hierarchy_level[wholes] = torch.maximum(hierarchy_level[wholes],
                                                            torch.tensor(i + 1, device=x.device))

            # Get hierarchy embeddings
            hierarchy_pe = self.hierarchy_embedding(hierarchy_level)
            encodings.append(hierarchy_pe)

        # Combine all encodings
        combined_pe = torch.cat(encodings, dim=-1)
        return self.combine(combined_pe)


class GrammaticalGTN(nn.Module):
    """
    Graph Transformer Network for grammatical construction analysis.
    """

    def __init__(
            self,
            node_dim,
            hidden_dim=128,
            num_layers=3,
            heads=4,
            dropout=0.1,
            edge_dim=None,
            max_nodes=100,
            use_structural_pe=True,
            use_part_whole=True
    ):
        super(GrammaticalGTN, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_structural_pe = use_structural_pe
        self.use_part_whole = use_part_whole

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Positional encoding
        if use_structural_pe:
            self.pe = StructuralPositionalEncoding(
                hidden_dim,
                max_nodes=max_nodes,
                use_centrality=True,
                use_part_whole=use_part_whole
            )

        # Graph Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                use_part_whole=use_part_whole
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        """
        Forward pass through the GTN.

        Args:
            data: PyTorch Geometric Data object containing graph information

        Returns:
            Node representations after graph transformer processing
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        part_of_mask = data.part_of_mask if hasattr(data, 'part_of_mask') else None
        batch = data.batch if hasattr(data, 'batch') else None

        # Initial projection
        x = self.input_proj(x)

        # Apply positional encoding
        if self.use_structural_pe:
            pe = self.pe(x, edge_index, part_of_mask, batch)
            x = x + pe

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, part_of_mask, batch)

        # Final projection
        x = self.output_proj(x)

        return x

    def encode_graph(self, data):
        """
        Encode the entire graph by pooling node representations.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Graph-level embedding
        """
        # Get node embeddings
        node_embeddings = self.forward(data)

        # Pool node embeddings to get graph embedding
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long,
                                                                      device=data.x.device)

        # Use mean pooling
        graph_embedding = scatter_mean(node_embeddings, batch, dim=0)

        return graph_embedding


# Simple classifier head for grammatical construction classification
class ConstructionClassifier(nn.Module):
    """
    A classifier for grammatical constructions built on top of the GTN.
    """

    def __init__(self, gtn, num_classes, hidden_dim=128):
        super(ConstructionClassifier, self).__init__()
        self.gtn = gtn

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        """
        Forward pass through the classifier.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Classification logits
        """
        # Get graph-level embedding
        graph_embedding = self.gtn.encode_graph(data)

        # Classify
        logits = self.classifier(graph_embedding)

        return logits


# Example usage
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data, Batch

    # Create a toy example
    x = torch.randn(10, 16)  # 10 nodes with 16 features
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 0, 2, 1, 3, 2, 5, 4, 7, 6, 9, 8]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1), 1)
    part_of_mask = torch.tensor([True, False, True, False, True, False, True, False, True, False, True, False])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, part_of_mask=part_of_mask)

    # Initialize the model
    gtn = GrammaticalGTN(
        node_dim=16,
        hidden_dim=32,
        num_layers=2,
        heads=4,
        edge_dim=1,
        use_structural_pe=True,
        use_part_whole=True
    )

    # Forward pass
    output = gtn(data)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test classifier
    classifier = ConstructionClassifier(gtn, num_classes=5)
    logits = classifier(data)

    print(f"Classification logits shape: {logits.shape}")