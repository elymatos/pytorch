import os
import pandas as pd
import numpy as np
import random
from datetime import datetime


def generate_sample_framenet_data(num_frames=50, num_lexical_units=200, output_dir="sample_data"):
    """
    Generate sample CSV files for FrameNet Brasil data.

    Args:
        num_frames: Number of frames to generate
        num_lexical_units: Number of lexical units to generate
        output_dir: Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # === Generate frames ===
    print(f"Generating {num_frames} frames...")

    # Frame names and definitions from generic semantic domains
    frame_domains = [
        "Motion", "Communication", "Perception", "Cognition", "Emotion",
        "Body", "Food", "Space", "Time", "Social", "Transaction", "Artifacts",
        "Natural_objects", "Causation", "Change", "Assessment", "Mental_activity",
        "Process", "Phenomenon", "Attributes", "Relations", "Events", "Substance",
        "Possession", "Exchange", "Environment", "Creation", "Destruction",
        "Competition", "Cooperation", "Leadership", "Achievement", "Punishment",
        "Reward", "Remembering", "Forgetting", "Learning", "Teaching", "Information",
        "Request", "Promise", "Judgment", "Opinion", "Belief", "Sound", "Vision",
        "Touch", "Taste", "Smell", "Health", "Medical"
    ]

    # Ensure we have enough frame domains
    if len(frame_domains) < num_frames:
        # Add numbered variations if we need more
        additional_frames = [f"{domain}_{i}" for domain in frame_domains for i in range(1, 5)]
        frame_domains.extend(additional_frames)

    # Randomly select a subset of frames
    selected_frames = random.sample(frame_domains, num_frames)

    # Create definitions for each frame
    frame_definitions = {
        "Motion": "This frame concerns the movement of a Theme from a Source to a Goal, with or without a Path.",
        "Communication": "A Communicator conveys a Message to an Addressee using a particular Medium.",
        "Perception": "A Perceiver perceives a Phenomenon using a particular sensory modality.",
        "Cognition": "This frame concerns a Cognizer's mental consideration or processing of some Content.",
        "Emotion": "An Experiencer has a particular emotional state or response to a Stimulus.",
    }

    # Generate generic definitions for frames without predefined definitions
    def generate_frame_definition(frame_name):
        if frame_name in frame_definitions:
            return frame_definitions[frame_name]

        # Create pattern-based definitions
        patterns = [
            "This frame involves entities that participate in the {0} process.",
            "In this frame, a {0}_agent engages with a {0}_patient in a specific context.",
            "The {0} frame describes situations in which participants interact through {0}-related activities.",
            "This frame characterizes scenarios where {0} is the primary activity or state.",
            "In the {0} frame, we find entities involved in processes related to {0} events or states."
        ]

        return random.choice(patterns).format(frame_name.lower())

    # Create frames dataframe
    frames_data = []
    for i, frame_name in enumerate(selected_frames):
        frame_id = i + 1  # Frame IDs start from 1
        frames_data.append({
            "frame_id": frame_id,
            "frame_name": frame_name,
            "frame_definition": generate_frame_definition(frame_name)
        })

    frames_df = pd.DataFrame(frames_data)

    # === Generate lexical units ===
    print(f"Generating {num_lexical_units} lexical units...")

    # Create lexical units for Portuguese
    portuguese_verbs = [
        "andar", "correr", "pular", "nadar", "voar", "dirigir", "navegar",
        "falar", "dizer", "conversar", "gritar", "sussurrar", "discutir",
        "ver", "olhar", "observar", "enxergar", "perceber", "notar",
        "pensar", "considerar", "ponderar", "refletir", "analisar",
        "sentir", "emocionar", "amar", "odiar", "gostar", "apreciar",
        "comer", "beber", "devorar", "saborear", "degustar",
        "comprar", "vender", "trocar", "negociar", "barganhar",
        "construir", "destruir", "criar", "desenvolver", "fabricar"
    ]

    portuguese_nouns = [
        "movimento", "deslocamento", "viagem", "passeio", "jornada",
        "conversa", "diálogo", "discurso", "mensagem", "comunicação",
        "visão", "observação", "olhar", "vista", "panorama",
        "pensamento", "ideia", "reflexão", "ponderação", "análise",
        "emoção", "sentimento", "amor", "ódio", "paixão",
        "corpo", "cabeça", "braço", "perna", "coração",
        "comida", "refeição", "prato", "alimento", "nutrição",
        "espaço", "lugar", "área", "região", "localização",
        "tempo", "momento", "instante", "período", "duração"
    ]

    portuguese_adjectives = [
        "rápido", "lento", "veloz", "ágil", "estático",
        "comunicativo", "eloquente", "falante", "calado", "expressivo",
        "observador", "atento", "perceptivo", "distraído", "detalhista",
        "pensativo", "reflexivo", "analítico", "racional", "intuitivo",
        "emotivo", "sensível", "apaixonado", "frio", "indiferente",
        "forte", "fraco", "robusto", "frágil", "resistente",
        "saboroso", "gostoso", "delicioso", "insípido", "amargo",
        "amplo", "estreito", "vasto", "limitado", "imenso",
        "breve", "longo", "eterno", "efêmero", "duradouro"
    ]

    # Define POS markers as in FrameNet
    pos_markers = {
        "verb": ".v",
        "noun": ".n",
        "adjective": ".a"
    }

    # Expand word lists if needed
    while len(portuguese_verbs) < num_lexical_units / 3:
        portuguese_verbs.extend([f"{verb}{i}" for verb in portuguese_verbs for i in range(1, 3)])

    while len(portuguese_nouns) < num_lexical_units / 3:
        portuguese_nouns.extend([f"{noun}{i}" for noun in portuguese_nouns for i in range(1, 3)])

    while len(portuguese_adjectives) < num_lexical_units / 3:
        portuguese_adjectives.extend([f"{adj}{i}" for adj in portuguese_adjectives for i in range(1, 3)])

    # Sample words
    selected_verbs = random.sample(portuguese_verbs, num_lexical_units // 3)
    selected_nouns = random.sample(portuguese_nouns, num_lexical_units // 3)
    selected_adjectives = random.sample(portuguese_adjectives, num_lexical_units - (num_lexical_units // 3) * 2)

    # Create sense descriptions
    def generate_sense_description(lemma, pos):
        if pos == "verb":
            patterns = [
                f"Ação de {lemma} com um propósito específico.",
                f"O processo de {lemma} realizado por um agente.",
                f"Ato de {lemma} em determinado contexto.",
                f"Realizar a ação de {lemma} de maneira intencional.",
                f"Engajar-se no ato de {lemma} com um objetivo."
            ]
        elif pos == "noun":
            patterns = [
                f"O {lemma} como entidade ou conceito abstrato.",
                f"Estado ou condição relacionada a {lemma}.",
                f"Objeto ou entidade classificada como {lemma}.",
                f"Manifestação concreta ou abstrata de {lemma}.",
                f"Representação ou instância de {lemma}."
            ]
        else:  # adjective
            patterns = [
                f"Qualidade ou característica de ser {lemma}.",
                f"Estado de apresentar-se como {lemma}.",
                f"Condição temporária ou permanente de ser {lemma}.",
                f"Propriedade distintiva de ser {lemma}.",
                f"Atributo ou traço definidor de ser {lemma}."
            ]

        return random.choice(patterns)

    # Generate lexical units
    lu_data = []
    lu_id_base = 100  # Start LU IDs from 100 to differentiate from frame IDs

    # Add verbs
    for i, verb in enumerate(selected_verbs):
        lu_id = lu_id_base + i
        lu_name = f"{verb}{pos_markers['verb']}"
        lu_data.append({
            "lu_id": lu_id,
            "lu_name": lu_name,
            "lemma": verb,
            "sense_description": generate_sense_description(verb, "verb")
        })

    # Add nouns
    lu_id_base = 200  # Different ID range for nouns
    for i, noun in enumerate(selected_nouns):
        lu_id = lu_id_base + i
        lu_name = f"{noun}{pos_markers['noun']}"
        lu_data.append({
            "lu_id": lu_id,
            "lu_name": lu_name,
            "lemma": noun,
            "sense_description": generate_sense_description(noun, "noun")
        })

    # Add adjectives
    lu_id_base = 300  # Different ID range for adjectives
    for i, adj in enumerate(selected_adjectives):
        lu_id = lu_id_base + i
        lu_name = f"{adj}{pos_markers['adjective']}"
        lu_data.append({
            "lu_id": lu_id,
            "lu_name": lu_name,
            "lemma": adj,
            "sense_description": generate_sense_description(adj, "adjective")
        })

    lu_df = pd.DataFrame(lu_data)

    # === Generate relations ===
    print("Generating relations...")

    # Define relation types
    relation_types = [
        "Inheritance",  # LU-to-Frame inheritance
        "Uses",  # Frame-to-Frame
        "Subframe",  # Frame-to-Frame
        "Precedes",  # Frame-to-Frame
        "Perspective_on",  # Frame-to-Frame
        "ReFraming_mapping"  # Frame-to-Frame
    ]

    relations_data = []

    # 1. Create Inheritance relations (LU to Frame)
    # Each LU inherits from one Frame
    frames_ids = frames_df["frame_id"].tolist()

    for _, lu_row in lu_df.iterrows():
        lu_id = lu_row["lu_id"]
        # Determine which frame this LU should connect to based on semantics

        # For demonstration, we'll just assign randomly, but you could implement
        # a more sophisticated mapping based on word meanings
        target_frame_id = random.choice(frames_ids)

        relations_data.append({
            "source_id": lu_id,
            "target_id": target_frame_id,
            "relation_type": "Inheritance"
        })

    # 2. Create Frame-to-Frame relations
    # We'll create a moderately connected graph
    # Each frame will have 1-3 relations to other frames

    for _, frame_row in frames_df.iterrows():
        frame_id = frame_row["frame_id"]

        # Number of relations for this frame (1-3)
        num_relations = random.randint(1, min(3, num_frames - 1))

        # Select target frames (excluding self)
        potential_targets = [fid for fid in frames_ids if fid != frame_id]
        target_frames = random.sample(potential_targets, min(num_relations, len(potential_targets)))

        for target_frame_id in target_frames:
            # Select a random relation type (excluding Inheritance which is for LU-to-Frame)
            relation_type = random.choice(relation_types[1:])

            relations_data.append({
                "source_id": frame_id,
                "target_id": target_frame_id,
                "relation_type": relation_type
            })

    relations_df = pd.DataFrame(relations_data)

    # === Save to CSV files ===
    frames_csv_path = os.path.join(output_dir, "frames.csv")
    lu_csv_path = os.path.join(output_dir, "lexical_units.csv")
    relations_csv_path = os.path.join(output_dir, "relations.csv")

    frames_df.to_csv(frames_csv_path, index=False)
    lu_df.to_csv(lu_csv_path, index=False)
    relations_df.to_csv(relations_csv_path, index=False)

    print(f"Generated {len(frames_df)} frames, {len(lu_df)} lexical units, and {len(relations_df)} relations.")
    print(f"CSV files saved to {output_dir}:")
    print(f"  - {frames_csv_path}")
    print(f"  - {lu_csv_path}")
    print(f"  - {relations_csv_path}")

    return {
        "frames_csv": frames_csv_path,
        "lexical_units_csv": lu_csv_path,
        "relations_csv": relations_csv_path,
        "num_frames": len(frames_df),
        "num_lexical_units": len(lu_df),
        "num_relations": len(relations_df)
    }


def display_sample_data(data_info):
    """
    Display sample data from the generated CSV files.

    Args:
        data_info: Dictionary with paths to CSV files
    """
    # Read the first few rows from each file
    frames_df = pd.read_csv(data_info["frames_csv"])
    lu_df = pd.read_csv(data_info["lexical_units_csv"])
    relations_df = pd.read_csv(data_info["relations_csv"])

    print("\n=== Sample Frames ===")
    print(frames_df.head())

    print("\n=== Sample Lexical Units ===")
    print(lu_df.head())

    print("\n=== Sample Relations ===")
    print(relations_df.head())

    # Print some statistics
    print("\n=== Data Statistics ===")
    print(f"Number of frames: {data_info['num_frames']}")
    print(f"Number of lexical units: {data_info['num_lexical_units']}")
    print(f"Number of relations: {data_info['num_relations']}")

    # Inheritance relations (LU to Frame)
    inheritance_relations = relations_df[relations_df["relation_type"] == "Inheritance"]
    print(f"Number of Inheritance relations (LU to Frame): {len(inheritance_relations)}")

    # Other relation types
    relation_counts = relations_df["relation_type"].value_counts()
    print("\nRelation type distribution:")
    for rel_type, count in relation_counts.items():
        print(f"  - {rel_type}: {count}")

    # Check graph connectivity
    all_nodes = set(frames_df["frame_id"]).union(set(lu_df["lu_id"]))
    nodes_in_relations = set(relations_df["source_id"]).union(set(relations_df["target_id"]))
    print(f"\nGraph coverage: {len(nodes_in_relations)}/{len(all_nodes)} nodes appear in relations")


if __name__ == "__main__":
    # Generate sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sample_data_{timestamp}"

    data_info = generate_sample_framenet_data(
        num_frames=30,  # Adjust as needed
        num_lexical_units=120,  # Adjust as needed
        output_dir=output_dir
    )

    # Display sample data
    display_sample_data(data_info)