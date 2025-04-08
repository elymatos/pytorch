# FrameNet Brasil R-GAT Link Prediction

This project implements a Relational Graph Attention Network (R-GAT) for link prediction in FrameNet Brasil's semantic graph. The model can predict inheritance relationships between lexical units (LUs) and frames, helping to extend the FrameNet Brasil database with new lexical entries.

## Overview

FrameNet Brasil is a semantic resource that represents linguistic knowledge through a network of:
- **Frames**: Semantic schemas that represent prototypical situations or concepts
- **Lexical Units (LUs)**: Words or phrases that evoke specific frames in particular contexts
- **Relations**: Connections between frames and LUs, including inheritance relationships

This project uses a graph neural network approach to model the complex relational structure of FrameNet Brasil, enabling automatic prediction of frame inheritance for new lexical units.

## Project Structure

- `framenet_rgat.py`: Core implementation of the R-GAT model and data processing
- `model_enhancements.py`: Advanced features and model extensions
- `generate_sample_data.py`: Utility to generate sample FrameNet data for testing
- `framenet_rgat_pipeline.py`: End-to-end pipeline that ties all components together

## Features

- **Graph-based Learning**: Leverages the relational structure of FrameNet Brasil using graph neural networks
- **Multiple Relation Types**: Handles various semantic relations between frames and lexical units
- **Advanced Feature Extraction**: 
  - BERT embeddings for frame definitions and LU sense descriptions
  - Part-of-speech features
  - Graph structural features
- **Comprehensive Evaluation**: Includes metrics, visualizations, and error analysis
- **Extensible Architecture**: Easily adapt to different languages or FrameNet versions

## Requirements

```
torch==2.0.1
torch-geometric==2.3.1
pandas
numpy
scikit-learn
matplotlib
seaborn
transformers
networkx
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/framenet-brasil-rgat.git
cd framenet-brasil-rgat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Using Sample Data

To run the pipeline with automatically generated sample data:

```bash
python framenet_rgat_pipeline.py
```

This will:
1. Generate sample FrameNet Brasil data
2. Process the data into a graph representation
3. Train an R-GAT model
4. Evaluate the model performance
5. Visualize the graph and predictions

### Using Your Own Data

To use your own FrameNet Brasil data:

```bash
python framenet_rgat_pipeline.py --no-sample-data \
  --frames-csv /path/to/frames.csv \
  --lexical-units-csv /path/to/lexical_units.csv \
  --relations-csv /path/to/relations.csv
```

### Advanced Usage

Enable all advanced features:

```bash
python framenet_rgat_pipeline.py \
  --enhanced-features \
  --multi-relation \
  --advanced-task \
  --epochs 300
```

## Data Format

The model expects three CSV files:

### frames.csv

```
frame_id,frame_name,frame_definition
1,Motion,"This frame concerns the movement of a Theme from a Source to a Goal, with or without a Path."
2,Communication,"A Communicator conveys a Message to an Addressee using a particular Medium."
3,Perception,"A Perceiver perceives a Phenomenon using a particular sensory modality."
```

### lexical_units.csv

```
lu_id,lu_name,lemma,sense_description
101,mover.v,mover,"To change position from one place to another"
102,falar.v,falar,"To express thoughts or feelings in spoken words"
103,ver.v,ver,"To perceive with the eyes"
```

### relations.csv

```
source_id,target_id,relation_type
101,1,Inheritance
102,2,Inheritance
103,3,Inheritance
1,2,Uses
2,3,Perspective_on
```

## Command Line Options

The pipeline script supports various command-line options:

```
usage: framenet_rgat_pipeline.py [-h] [--no-sample-data] [--frames-csv FRAMES_CSV]
                               [--lexical-units-csv LEXICAL_UNITS_CSV]
                               [--relations-csv RELATIONS_CSV] [--num-frames NUM_FRAMES]
                               [--num-lexical-units NUM_LEXICAL_UNITS]
                               [--enhanced-features] [--no-bert] [--no-pos]
                               [--no-graph-features] [--no-visualize]
                               [--hidden-channels HIDDEN_CHANNELS]
                               [--out-channels OUT_CHANNELS]
                               [--attention-heads ATTENTION_HEADS] [--dropout DROPOUT]
                               [--multi-relation]
                               [--relation-embedding-dim RELATION_EMBEDDING_DIM]
                               [--advanced-task] [--epochs EPOCHS]
                               [--base-dir BASE_DIR]

FrameNet Brasil R-GAT Pipeline
```

## Model Architecture

The R-GAT model architecture consists of:

1. **Input Layer**: Takes node features and graph structure
2. **R-GAT Layers**: Apply relational graph attention to learn node representations
3. **Link Prediction Layer**: Predicts the likelihood of connections between nodes

The Multi-Relation variant enhances the model with:
- Relation-specific embeddings
- Relation-aware attention mechanisms
- Context-sensitive link prediction

## Results and Evaluation

The pipeline produces comprehensive evaluation results in the experiment directory:

- **Metrics**: AUC, precision, recall, F1 score
- **Visualizations**:
  - Graph structure
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - t-SNE visualization of node embeddings
  - Top frame predictions for example lexical units

## Extending the Model

### Adding New Features

To incorporate additional features:

1. Modify the `AdvancedFeatureExtractor` class in `model_enhancements.py`
2. Add your feature extraction method
3. Update the `extract_node_features` method to include your new features

### Customizing for Other Languages

The model can be adapted to other FrameNet projects by:

1. Adjusting the data loading to match your format
2. Modifying the feature extraction for language-specific characteristics
3. Updating the BERT model to use a multilingual or language-specific model

## Citation

If you use this code in your research, please cite:

```
@software{framenet_brasil_rgat,
  author = {Your Name},
  title = {FrameNet Brasil R-GAT Link Prediction},
  year = {2025},
  url = {https://github.com/yourusername/framenet-brasil-rgat}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FrameNet Brasil project for the semantic framework
- PyTorch Geometric for the graph neural network implementation