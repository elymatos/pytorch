import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import our modules
# Assuming these files are in the same directory
from generate_sample_data import generate_sample_framenet_data, display_sample_data
from framenet_rgat import FrameNetDataProcessor, RGAT, LinkPredictionTask
from model_enhancements import (
    AdvancedFeatureExtractor,
    EnhancedFrameNetDataProcessor,
    MultiRelationRGAT,
    AdvancedLinkPredictionTask,
    evaluate_model,
    visualize_graph
)


def create_experiment_directory(base_dir="experiments"):
    """Create a timestamped directory for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ["data", "models", "results", "visualizations"]:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    return experiment_dir


def save_config(config, experiment_dir):
    """Save experiment configuration to JSON file"""
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {config_path}")
    return config_path


def run_pipeline(config):
    """
    Run the complete R-GAT pipeline for FrameNet Brasil link prediction

    Args:
        config: Dictionary with configuration parameters
    """
    print("\n=== Starting FrameNet Brasil R-GAT Pipeline ===\n")

    # Create experiment directory
    experiment_dir = create_experiment_directory(config.get("base_dir", "experiments"))
    print(f"Experiment directory: {experiment_dir}")

    # Save configuration
    save_config(config, experiment_dir)

    # Step 1: Generate or load data
    data_dir = os.path.join(experiment_dir, "data")

    if config.get("generate_sample_data", True):
        print("\n=== Generating Sample Data ===\n")

        data_info = generate_sample_framenet_data(
            num_frames=config.get("num_frames", 30),
            num_lexical_units=config.get("num_lexical_units", 120),
            output_dir=data_dir
        )

        # Display sample data
        if config.get("display_sample_data", True):
            display_sample_data(data_info)

        frames_csv = data_info["frames_csv"]
        lexical_units_csv = data_info["lexical_units_csv"]
        relations_csv = data_info["relations_csv"]
    else:
        # Use existing data
        frames_csv = config.get("frames_csv", "frames.csv")
        lexical_units_csv = config.get("lexical_units_csv", "lexical_units.csv")
        relations_csv = config.get("relations_csv", "relations.csv")

    # Step 2: Process data
    print("\n=== Processing Data ===\n")

    use_enhanced_features = config.get("use_enhanced_features", False)

    if use_enhanced_features:
        print("Using enhanced features with BERT, POS, and graph features")
        processor = EnhancedFrameNetDataProcessor(
            frames_csv=frames_csv,
            lexical_units_csv=lexical_units_csv,
            relations_csv=relations_csv,
            use_bert=config.get("use_bert", True),
            use_pos=config.get("use_pos", True),
            use_graph_features=config.get("use_graph_features", True)
        )
    else:
        print("Using basic features")
        processor = FrameNetDataProcessor(
            frames_csv=frames_csv,
            lexical_units_csv=lexical_units_csv,
            relations_csv=relations_csv
        )

    data, num_relations = processor.process()

    # Step 3: Visualize graph (optional)
    if config.get("visualize_graph", True):
        print("\n=== Visualizing Graph ===\n")

        visualization_dir = os.path.join(experiment_dir, "visualizations")
        graph = visualize_graph(data, processor, output_dir=visualization_dir)
        print(f"Graph visualizations saved to {visualization_dir}")

    # Step 4: Create model
    print("\n=== Creating Model ===\n")

    in_channels = data.x.size(1)
    hidden_channels = config.get("hidden_channels", 128)
    out_channels = config.get("out_channels", 64)
    heads = config.get("attention_heads", 8)
    dropout = config.get("dropout", 0.3)

    use_multi_relation = config.get("use_multi_relation", False)

    if use_multi_relation:
        print("Using Multi-Relation RGAT model")
        model = MultiRelationRGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            heads=heads,
            dropout=dropout,
            relation_embedding_dim=config.get("relation_embedding_dim", 32)
        )
    else:
        print("Using standard RGAT model")
        model = RGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            heads=heads,
            dropout=dropout
        )

    # Step 5: Set up link prediction task
    print("\n=== Setting Up Link Prediction Task ===\n")

    use_advanced_task = config.get("use_advanced_task", False)

    if use_advanced_task:
        print("Using Advanced Link Prediction Task")
        task = AdvancedLinkPredictionTask(model)
        relation_data = task.prepare_data_for_relation_prediction(data)
    else:
        print("Using standard Link Prediction Task")
        task = LinkPredictionTask(model)

    link_data = task.prepare_data_for_link_prediction(data)

    # Step 6: Train model
    print("\n=== Training Model ===\n")

    epochs = config.get("epochs", 200)

    if use_advanced_task and use_multi_relation:
        print(f"Training with relation awareness for {epochs} epochs")
        task.train_with_relations(data, relation_data, epochs=epochs)
    else:
        print(f"Standard training for {epochs} epochs")
        task.train(data, link_data, epochs=epochs)

    # Step 7: Evaluate model
    print("\n=== Evaluating Model ===\n")

    results_dir = os.path.join(experiment_dir, "results")
    metrics = evaluate_model(model, data, link_data, processor, output_dir=results_dir)

    # Step 8: Save model
    model_path = os.path.join(experiment_dir, "models", "framenet_rgat_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Step 9: Make predictions for new lexical units
    print("\n=== Making Predictions for New Lexical Units ===\n")

    # For demonstration, we'll use existing LUs as if they were new
    # In a real scenario, you would have new LUs to predict
    lu_indices = torch.where(data.node_type == 1)[0][:5]  # First 5 LUs as examples

    if use_advanced_task and use_multi_relation:
        # Find the index of the 'Inheritance' relation type
        # This should be adjusted based on your actual relation encodings
        inheritance_type = 0  # Assuming the first relation type is Inheritance

        predictions = task.predict_lu_frame_links_with_relations(
            data, lu_indices, inheritance_type, top_k=5)
    else:
        predictions = task.predict_lu_frame_links(data, lu_indices, top_k=5)

    # Print predictions
    print("\nPredicted frame links for example lexical units:")
    for lu_idx, frame_preds in predictions.items():
        lu_id, lu_name, _ = processor.node_to_id_mapping[lu_idx]
        print(f"\nTop frame predictions for LU '{lu_name}' (ID: {lu_id}):")
        for frame_idx, score in frame_preds:
            frame_id, frame_name, _ = processor.node_to_id_mapping[frame_idx]
            print(f"  - {frame_name} (ID: {frame_id}): {score:.4f}")

    # Step 10: Save results summary
    results_summary = {
        "experiment_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "config": config
    }

    summary_path = os.path.join(experiment_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=4)

    print(f"\nResults summary saved to {summary_path}")

    print("\n=== Pipeline Completed Successfully ===\n")
    return experiment_dir, metrics


def get_default_config():
    """
    Get default configuration for the pipeline
    """
    return {
        # Data settings
        "generate_sample_data": True,
        "num_frames": 30,
        "num_lexical_units": 120,
        "display_sample_data": True,

        # Feature settings
        "use_enhanced_features": False,
        "use_bert": True,
        "use_pos": True,
        "use_graph_features": True,

        # Visualization settings
        "visualize_graph": True,

        # Model settings
        "hidden_channels": 128,
        "out_channels": 64,
        "attention_heads": 8,
        "dropout": 0.3,
        "use_multi_relation": False,
        "relation_embedding_dim": 32,

        # Training settings
        "use_advanced_task": False,
        "epochs": 200,

        # Experiment settings
        "base_dir": "experiments"
    }


def parse_arguments():
    """
    Parse command-line arguments for the pipeline
    """
    parser = argparse.ArgumentParser(description="FrameNet Brasil R-GAT Pipeline")

    # Data settings
    parser.add_argument("--no-sample-data", action="store_true",
                        help="Use existing data instead of generating sample data")
    parser.add_argument("--frames-csv", type=str, default="frames.csv",
                        help="Path to frames CSV file (only used if --no-sample-data is set)")
    parser.add_argument("--lexical-units-csv", type=str, default="lexical_units.csv",
                        help="Path to lexical units CSV file (only used if --no-sample-data is set)")
    parser.add_argument("--relations-csv", type=str, default="relations.csv",
                        help="Path to relations CSV file (only used if --no-sample-data is set)")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Number of frames to generate (only used if generating sample data)")
    parser.add_argument("--num-lexical-units", type=int, default=120,
                        help="Number of lexical units to generate (only used if generating sample data)")

    # Feature settings
    parser.add_argument("--enhanced-features", action="store_true",
                        help="Use enhanced features (BERT, POS, graph features)")
    parser.add_argument("--no-bert", action="store_true",
                        help="Disable BERT features (only used if --enhanced-features is set)")
    parser.add_argument("--no-pos", action="store_true",
                        help="Disable POS features (only used if --enhanced-features is set)")
    parser.add_argument("--no-graph-features", action="store_true",
                        help="Disable graph features (only used if --enhanced-features is set)")

    # Visualization settings
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable graph visualization")

    # Model settings
    parser.add_argument("--hidden-channels", type=int, default=128,
                        help="Number of hidden channels in the model")
    parser.add_argument("--out-channels", type=int, default=64,
                        help="Number of output channels in the model")
    parser.add_argument("--attention-heads", type=int, default=8,
                        help="Number of attention heads in the model")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--multi-relation", action="store_true",
                        help="Use multi-relation RGAT model")
    parser.add_argument("--relation-embedding-dim", type=int, default=32,
                        help="Dimension of relation embeddings (only used if --multi-relation is set)")

    # Training settings
    parser.add_argument("--advanced-task", action="store_true",
                        help="Use advanced link prediction task")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")

    # Experiment settings
    parser.add_argument("--base-dir", type=str, default="experiments",
                        help="Base directory for experiment results")

    args = parser.parse_args()

    # Convert arguments to configuration dictionary
    config = get_default_config()

    # Update config with command-line arguments
    config["generate_sample_data"] = not args.no_sample_data
    if not args.no_sample_data:
        config["num_frames"] = args.num_frames
        config["num_lexical_units"] = args.num_lexical_units
    else:
        config["frames_csv"] = args.frames_csv
        config["lexical_units_csv"] = args.lexical_units_csv
        config["relations_csv"] = args.relations_csv

    config["use_enhanced_features"] = args.enhanced_features
    if args.enhanced_features:
        config["use_bert"] = not args.no_bert
        config["use_pos"] = not args.no_pos
        config["use_graph_features"] = not args.no_graph_features

    config["visualize_graph"] = not args.no_visualize

    config["hidden_channels"] = args.hidden_channels
    config["out_channels"] = args.out_channels
    config["attention_heads"] = args.attention_heads
    config["dropout"] = args.dropout
    config["use_multi_relation"] = args.multi_relation
    config["relation_embedding_dim"] = args.relation_embedding_dim

    config["use_advanced_task"] = args.advanced_task
    config["epochs"] = args.epochs

    config["base_dir"] = args.base_dir

    return config


if __name__ == "__main__":
    # Parse command-line arguments
    config = parse_arguments()

    # Run pipeline
    experiment_dir, metrics = run_pipeline(config)

    print(f"\nExperiment results available in {experiment_dir}")
    print(f"Test AUC: {metrics.get('test_auc', 'N/A')}")
    print(f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"PR AUC: {metrics.get('pr_auc', 'N/A')}")