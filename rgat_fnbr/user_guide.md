# Usage Guide for FrameNet Brasil R-GAT Model

This guide will help you use the R-GAT (Relational Graph Attention Network) model for link prediction in the FrameNet Brasil semantic graph.

## Installation Requirements

First, make sure you have the necessary libraries installed:

```bash
pip install torch==2.0.1
pip install torch-geometric==2.3.1
pip install pandas numpy scikit-learn
```

Note: The exact versions might need to be adjusted based on your system configuration.

## Workflow Overview

1. Prepare your data in the required CSV format
2. Load and process the data
3. Train the R-GAT model
4. Evaluate the model
5. Predict frame links for new lexical units

## Step-by-Step Usage

### 1. Prepare Your Data

Follow the format described in the CSV File Format document. Ensure you have:
- frames.csv
- lexical_units.csv
- relations.csv

### 2. Run the Model

The simplest way to use the model is with the default settings:

```python
from framenet_rgat import main

# This will load the data, train the model, and evaluate it
main()
```

### 3. Custom Usage for Specific Tasks

If you want more control over the process, you can use the individual components:

```python
from framenet_rgat import FrameNetDataProcessor, RGAT, LinkPredictionTask
import torch

# Set file paths
frames_csv = "path/to/frames.csv"
lexical_units_csv = "path/to/lexical_units.csv"
relations_csv = "path/to/relations.csv"

# Process data
processor = FrameNetDataProcessor(frames_csv, lexical_units_csv, relations_csv)
data, num_relations = processor.process()

# Create and configure the model
in_channels = data.x.size(1)
hidden_channels = 128  # You can adjust this
out_channels = 64      # You can adjust this
model = RGAT(in_channels, hidden_channels, out_channels, num_relations, heads=4)

# Set up the task handler
task = LinkPredictionTask(model)
link_data = task.prepare_data_for_link_prediction(data)

# Train with custom parameters
task.train(data, link_data, epochs=300)  # Increase epochs for better results

# Evaluate
test_auc = task.evaluate(data, link_data, 'test')
print(f'Test AUC: {test_auc:.4f}')
```

### 4. Predicting Links for New Lexical Units

To predict frames for new lexical units:

```python
# First, add your new lexical units to lexical_units.csv
# Then reload the data with the updated CSV

# Get the indices of the new lexical units
# For example, if your new lexical units have the following IDs:
new_lu_ids = [1001, 1002, 1003]  

# Convert IDs to indices in the graph
new_lu_indices = torch.tensor([
    processor.id_to_node_mapping[lu_id] for lu_id in new_lu_ids
])

# Predict top 5 frames for each new lexical unit
predictions = task.predict_lu_frame_links(data, new_lu_indices, top_k=5)

# Print predictions
for lu_idx, frame_preds in predictions.items():
    lu_id, lu_name, _ = processor.node_to_id_mapping[lu_idx]
    print(f"\nTop frame predictions for LU '{lu_name}' (ID: {lu_id}):")
    for frame_idx, score in frame_preds:
        frame_id, frame_name, _ = processor.node_to_id_mapping[frame_idx]
        print(f"  - {frame_name} (ID: {frame_id}): {score:.4f}")
```

## Advanced Usage

### Using Pre-trained Embeddings

For better performance, you can use pre-trained embeddings (like BERT or Word2Vec) for frame definitions and lexical unit sense descriptions:

1. Modify the `extract_node_features` method in the `FrameNetDataProcessor` class
2. Replace the random initialization with embeddings from your pre-trained model

Example with BERT:

```python
from transformers import BertModel, BertTokenizer

def extract_node_features(self, frames_df, lu_df):
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Get embeddings for each node
    node_features = []
    for idx in range(len(self.node_to_id_mapping)):
        node_id, node_name, node_type = self.node_to_id_mapping[idx]
        
        # Get text to embed based on node type
        if node_type == 0:  # Frame
            frame_row = frames_df[frames_df['frame_id'] == node_id].iloc[0]
            text = frame_row['frame_definition']
        else:  # Lexical Unit
            lu_row = lu_df[lu_df['lu_id'] == node_id].iloc[0]
            text = lu_row['sense_description']
        
        # Get BERT embedding
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach()
        
        node_features.append(embedding)
    
    # Stack into tensor
    return torch.stack(node_features)
```

### Hyperparameter Tuning

The model performance can be improved by tuning hyperparameters:

```python
# Example of hyperparameter configurations to try
configs = [
    {'hidden_channels': 64, 'out_channels': 32, 'heads': 4, 'dropout': 0.2},
    {'hidden_channels': 128, 'out_channels': 64, 'heads': 8, 'dropout': 0.3},
    {'hidden_channels': 256, 'out_channels': 128, 'heads': 4, 'dropout': 0.4}
]

best_auc = 0
best_config = None

for config in configs:
    # Create model with this config
    model = RGAT(
        in_channels=data.x.size(1), 
        hidden_channels=config['hidden_channels'], 
        out_channels=config['out_channels'], 
        num_relations=num_relations,
        heads=config['heads'],
        dropout=config['dropout']
    )
    
    # Train and evaluate
    task = LinkPredictionTask(model)
    link_data = task.prepare_data_for_link_prediction(data)
    task.train(data, link_data, epochs=200)
    auc = task.evaluate(data, link_data, 'val')
    
    print(f"Config {config} achieved AUC: {auc:.4f}")
    
    # Track best config
    if auc > best_auc:
        best_auc = auc
        best_config = config

print(f"Best config: {best_config} with AUC: {best_auc:.4f}")
```

## Troubleshooting

1. **Memory Issues**: If you encounter memory issues with large graphs, try:
   - Reducing batch size
   - Using fewer attention heads
   - Reducing hidden dimensions

2. **Poor Performance**: If the model performance is not satisfactory:
   - Increase the number of training epochs
   - Use pre-trained embeddings instead of random initialization
   - Add more features to the nodes