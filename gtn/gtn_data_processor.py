import json
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Union, Optional


class GrammaticalGraphProcessor:
    """
    A processor for grammatical construction graphs that loads JSON data
    and converts it into PyTorch Geometric data structures for GTN processing.
    """

    def __init__(self, feature_dim: int = 64):
        """
        Initialize the graph processor.

        Args:
            feature_dim: Dimension for node features after embedding
        """
        self.feature_dim = feature_dim
        # Maps for converting string values to indices
        self.node_type_map = {}
        self.edge_type_map = {}

    def load_from_json(self, json_file: str) -> List[Data]:
        """
        Load graphs from a JSON file.

        Args:
            json_file: Path to the JSON file containing graph data

        Returns:
            List of PyTorch Geometric Data objects
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # Multiple graphs in the file
            return [self._process_single_graph(graph_data) for graph_data in data]
        else:
            # Single graph in the file
            return [self._process_single_graph(data)]

    def load_from_json_str(self, json_str: str) -> List[Data]:
        """
        Load graphs from a JSON string.

        Args:
            json_str: JSON string containing graph data

        Returns:
            List of PyTorch Geometric Data objects
        """
        data = json.loads(json_str)

        if isinstance(data, list):
            # Multiple graphs in the string
            return [self._process_single_graph(graph_data) for graph_data in data]
        else:
            # Single graph in the string
            return [self._process_single_graph(data)]

    def _process_single_graph(self, graph_data: Dict) -> Data:
        """
        Process a single graph from parsed JSON data.

        Args:
            graph_data: Dictionary containing the graph data

        Returns:
            PyTorch Geometric Data object
        """
        # Extract nodes and edges
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        meta = graph_data.get('meta', {})

        # Create node ID to index mapping
        node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}

        # Process nodes
        x = self._process_nodes(nodes)

        # Process edges
        edge_index, edge_attr, part_of_mask = self._process_edges(edges, node_id_to_idx)

        # Create PyTorch Geometric Data object
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            part_of_mask=part_of_mask,
            num_nodes=len(nodes)
        )

        # Add metadata as additional attributes
        for key, value in meta.items():
            graph[f'meta_{key}'] = value

        return graph

    def _process_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        """
        Process nodes from the JSON data.

        Args:
            nodes: List of node dictionaries

        Returns:
            Node feature tensor
        """
        # Initialize feature tensor
        x = torch.zeros((len(nodes), self.feature_dim))

        for i, node in enumerate(nodes):
            # Process node type
            node_type = node.get('type', 'unknown')
            if node_type not in self.node_type_map:
                self.node_type_map[node_type] = len(self.node_type_map)
            node_type_idx = self.node_type_map[node_type]

            # Get text embedding (in a real implementation, you might use a pretrained model)
            # Here we're just using a simple hash function as a placeholder
            text = node.get('text', '')
            text_hash = sum(ord(c) for c in text) % 100  # Simple hash for demonstration

            # Get other features
            features = node.get('features', {})

            # Create a simple feature vector - in a real implementation you'd have more sophisticated features
            # The first few indices represent node type (one-hot encoded)
            x[i, node_type_idx % 10] = 1.0

            # Add text hash to the feature
            x[i, 10] = text_hash / 100.0

            # Add POS tag if available
            pos = features.get('pos', 'NONE')
            pos_hash = sum(ord(c) for c in pos) % 10
            x[i, 11] = pos_hash / 10.0

            # You would add more complex feature processing here based on your specific needs

        return x

    def _process_edges(self, edges: List[Dict], node_id_to_idx: Dict[str, int]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process edges from the JSON data.

        Args:
            edges: List of edge dictionaries
            node_id_to_idx: Mapping from node IDs to indices

        Returns:
            Tuple of (edge_index, edge_attr, part_of_mask) tensors
        """
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1)), torch.zeros((0,), dtype=torch.bool)

        # Prepare edge data structures
        edge_indices = []
        edge_features = []
        part_of_flags = []

        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')

            if source_id not in node_id_to_idx or target_id not in node_id_to_idx:
                continue  # Skip edges referencing unknown nodes

            source_idx = node_id_to_idx[source_id]
            target_idx = node_id_to_idx[target_id]

            # Add edge to edge list (PyTorch Geometric uses source, target convention)
            edge_indices.append([source_idx, target_idx])

            # Process edge type
            edge_type = edge.get('type', 'unknown')
            if edge_type not in self.edge_type_map:
                self.edge_type_map[edge_type] = len(self.edge_type_map)
            edge_type_idx = self.edge_type_map[edge_type]

            # Flag if this is a part-of relationship
            is_part_of = edge_type == 'part_of'
            part_of_flags.append(is_part_of)

            # Create a simple edge feature vector
            edge_feature = [edge_type_idx]

            # You could add more edge features from the 'features' field if needed
            edge_features.append(edge_feature)

        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()  # Transpose to get 2 x num_edges
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        part_of_mask = torch.tensor(part_of_flags, dtype=torch.bool)

        return edge_index, edge_attr, part_of_mask

    def batch_graphs(self, graphs: List[Data]) -> Batch:
        """
        Batch multiple graphs into a single batch.

        Args:
            graphs: List of PyTorch Geometric Data objects

        Returns:
            Batched graph
        """
        return Batch.from_data_list(graphs)


# Example usage:
if __name__ == "__main__":
    # Example JSON string (you would typically load this from a file)
    example_json = '''
    {
      "nodes": [
        {
          "id": "n1",
          "type": "construction",
          "text": "subject verb object",
          "features": {
            "pos": "CONSTRUCTION"
          }
        },
        {
          "id": "n2",
          "type": "word",
          "text": "The",
          "features": {
            "pos": "DET"
          }
        },
        {
          "id": "n3",
          "type": "word",
          "text": "cat",
          "features": {
            "pos": "NOUN"
          }
        },
        {
          "id": "n4",
          "type": "word",
          "text": "chased",
          "features": {
            "pos": "VERB"
          }
        },
        {
          "id": "n5",
          "type": "word",
          "text": "the",
          "features": {
            "pos": "DET"
          }
        },
        {
          "id": "n6",
          "type": "word",
          "text": "mouse",
          "features": {
            "pos": "NOUN"
          }
        }
      ],
      "edges": [
        {
          "source": "n2",
          "target": "n1",
          "type": "part_of",
          "label": "subject_determiner"
        },
        {
          "source": "n3",
          "target": "n1",
          "type": "part_of",
          "label": "subject_noun"
        },
        {
          "source": "n4",
          "target": "n1",
          "type": "part_of",
          "label": "verb"
        },
        {
          "source": "n5",
          "target": "n1",
          "type": "part_of",
          "label": "object_determiner"
        },
        {
          "source": "n6",
          "target": "n1",
          "type": "part_of",
          "label": "object_noun"
        }
      ],
      "meta": {
        "source_text": "The cat chased the mouse.",
        "language": "en"
      }
    }
    '''

    # Initialize processor
    processor = GrammaticalGraphProcessor(feature_dim=64)

    # Process the example JSON
    graphs = processor.load_from_json_str(example_json)

    # Print some information about the processed graph
    graph = graphs[0]
    print(f"Graph has {graph.num_nodes} nodes and {graph.edge_index.size(1)} edges")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge attributes shape: {graph.edge_attr.shape}")
    print(f"Part-of mask shape: {graph.part_of_mask.shape}")
    print(f"Number of part-of relationships: {graph.part_of_mask.sum().item()}")