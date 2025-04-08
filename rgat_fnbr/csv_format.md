# CSV File Format for FrameNet Brasil R-GAT Model

To use the R-GAT model for FrameNet Brasil link prediction, you need to prepare your data in three CSV files:

## 1. frames.csv

This file contains information about semantic frames in FrameNet Brasil.

| Column | Description |
|--------|-------------|
| frame_id | Unique identifier for the frame |
| frame_name | Name of the frame |
| frame_definition | Full definition of the frame |

Example:
```
frame_id,frame_name,frame_definition
1,Motion,"This frame concerns the movement of a Theme from a Source to a Goal, with or without a Path."
2,Communication,"A Communicator conveys a Message to an Addressee using a particular Medium."
3,Perception,"A Perceiver perceives a Phenomenon using a particular sensory modality."
```

## 2. lexical_units.csv

This file contains information about lexical units in FrameNet Brasil.

| Column | Description |
|--------|-------------|
| lu_id | Unique identifier for the lexical unit |
| lu_name | Name of the lexical unit |
| lemma | The base form of the lexical unit |
| sense_description | Description of the sense of the lexical unit |

Example:
```
lu_id,lu_name,lemma,sense_description
101,mover.v,mover,"To change position from one place to another"
102,falar.v,falar,"To express thoughts or feelings in spoken words"
103,ver.v,ver,"To perceive with the eyes"
```

## 3. relations.csv

This file contains the relations between entities (frames and lexical units) in the network.

| Column | Description |
|--------|-------------|
| source_id | ID of the source entity (can be a frame or LU) |
| target_id | ID of the target entity (can be a frame or LU) |
| relation_type | Type of relation (e.g., "Inheritance", "Uses", "Perspective_on") |

Example:
```
source_id,target_id,relation_type
101,1,Inheritance
102,2,Inheritance
103,3,Inheritance
1,2,Uses
2,3,Perspective_on
```

## Important Notes:

1. The IDs in `source_id` and `target_id` should match the IDs defined in either `frame_id` or `lu_id`.
2. For link prediction of LU-to-Frame inheritance relations, make sure the `relations.csv` file includes these relationships with the appropriate relation_type (e.g., "Inheritance").
3. Each lexical unit should have at least one relation to a frame in the training data for the model to learn effectively.
4. While the model is designed to handle missing data, having more complete data will yield better results.
5. The ID systems for frames and lexical units should not overlap to avoid confusion in the model.

## Handling Additional Data:

If you have additional data about frames or lexical units that you want to incorporate:

1. You can add additional columns to `frames.csv` or `lexical_units.csv`.
2. You'll need to modify the `FrameNetDataProcessor` class in the main code to extract features from these additional columns.
3. For the best results, consider using pre-trained embeddings for frame definitions and lexical unit sense descriptions.