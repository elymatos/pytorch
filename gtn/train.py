import os
import json
import torch
import argparse
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our custom modules
from gtn_data_processor import GrammaticalGraphProcessor
from gtn_model import GrammaticalGTN, ConstructionClassifier


def train(model, train_loader, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        device: Device to run training on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)

        # Forward pass
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    """
    Validate the model.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on

    Returns:
        Tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(device)

            # Forward pass
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, batch.y)

            # Calculate accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def prepare_data(data_dir, feature_dim=64, train_ratio=0.8, batch_size=32):
    """
    Prepare data for training.

    Args:
        data_dir: Directory containing JSON data files
        feature_dim: Dimension for node features
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for DataLoader

    Returns:
        Tuple of (train_loader, val_loader, num_classes, edge_dim)
    """
    # Initialize the graph processor
    processor = GrammaticalGraphProcessor(feature_dim=feature_dim)

    # Load all JSON files in the directory
    all_graphs = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            graphs = processor.load_from_json(file_path)
            all_graphs.extend(graphs)

    if not all_graphs:
        raise ValueError(f"No valid graph data found in {data_dir}")

    print(f"Loaded {len(all_graphs)} graphs for training")

    # Add labels to the graphs
    label_set = set()
    for graph in all_graphs:
        # Extract construction type from the metadata
        label = graph.meta_construction_type if hasattr(graph, 'meta_construction_type') else (
            graph.meta_construction_type if hasattr(graph, 'meta_construction_type') else 0)
        # Create a scalar tensor rather than a 1-element tensor
        graph.y = torch.tensor(label, dtype=torch.long)
        label_set.add(label)

    # Add labels to the graphs (this assumes that label information is present in the JSON)
    # In a real scenario, you might need a more sophisticated way to assign labels
    # label_set = set()
    # for graph in all_graphs:
    #     # Try to extract label from metadata or another source
    #     # This is just a placeholder - replace with your actual logic
    #     label = graph.meta_construction_type if hasattr(graph, 'meta_construction_type') else 0
    #     graph.y = torch.tensor([label], dtype=torch.long)
    #     label_set.add(label)

    num_classes = len(label_set)
    print(f"Found {num_classes} different construction classes")

    # Determine edge feature dimension if edges have features
    edge_dim = None
    for graph in all_graphs:
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_dim = graph.edge_attr.size(1)
            break

    # Split data into training and validation sets
    num_train = int(len(all_graphs) * train_ratio)
    indices = torch.randperm(len(all_graphs)).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_graphs = [all_graphs[i] for i in train_indices]
    val_graphs = [all_graphs[i] for i in val_indices]

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes, edge_dim


def visualize_training(train_losses, val_losses, val_accuracies, save_dir):
    """
    Visualize training progress.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()


def analyze_attention(model, data_loader, device, save_dir, num_samples=5):
    """
    Analyze and visualize attention patterns in the model.

    Args:
        model: Trained model
        data_loader: DataLoader containing data samples
        device: Device to run inference on
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Get a few samples
    samples = []
    for batch in data_loader:
        batch = batch.to(device)
        samples.extend(batch.to_data_list()[:num_samples - len(samples)])
        if len(samples) >= num_samples:
            break

    for i, sample in enumerate(samples):
        # Get attention weights for this sample
        # Note: This would require modifying the model to return attention weights
        # This is just a placeholder for the actual implementation

        # Visualize the graph structure
        plt.figure(figsize=(10, 8))
        # Create a simple visualization of the graph
        # In a real implementation, you would use a more sophisticated network visualization library

        # Get node positions (this is a simplification)
        pos = {}
        num_nodes = sample.x.size(0)
        for j in range(num_nodes):
            theta = 2 * np.pi * j / num_nodes
            pos[j] = (np.cos(theta), np.sin(theta))

        # Draw nodes
        for j in range(num_nodes):
            plt.plot(pos[j][0], pos[j][1], 'o', markersize=10, color='blue')
            plt.text(pos[j][0] * 1.1, pos[j][1] * 1.1, f'Node {j}')

        # Draw edges
        edge_index = sample.edge_index.cpu().numpy()
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            plt.plot([pos[src][0], pos[dst][0]], [pos[src][1], pos[dst][1]], 'k-')

        plt.title(f'Graph Structure - Sample {i + 1}')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'graph_structure_{i + 1}.png'))
        plt.close()


def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, num_classes, edge_dim = prepare_data(
        args.data_dir,
        feature_dim=args.feature_dim,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size
    )

    # Initialize model
    gtn = GrammaticalGTN(
        node_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout,
        edge_dim=edge_dim,
        max_nodes=args.max_nodes,
        use_structural_pe=args.use_structural_pe,
        use_part_whole=args.use_part_whole
    ).to(device)

    model = ConstructionClassifier(
        gtn=gtn,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Set up optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")

    # Visualize training progress
    visualize_training(train_losses, val_losses, val_accuracies, args.output_dir)

    # Analyze attention patterns
    analyze_attention(model, val_loader, device, os.path.join(args.output_dir, 'attention_viz'))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTN for Grammatical Construction Analysis")

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing JSON data files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
    parser.add_argument('--feature_dim', type=int, default=64, help='Dimension of node features')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GTN layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_nodes', type=int, default=100, help='Maximum number of nodes per graph')
    parser.add_argument('--use_structural_pe', action='store_true', help='Use structural positional encoding')
    parser.add_argument('--use_part_whole', action='store_true', help='Use part-whole relationship modeling')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Train model
    model = main(args)