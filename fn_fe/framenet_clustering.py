import pandas as pd
import networkx as nx
import csv

# --- Configuration ---
# File paths for your uploaded data
FE_ALL_FILE = 'fnbr_frameelement_all.csv'
FE_RELATIONS_FILE = 'fnbr_fe_relations_frames.csv'
OUTPUT_CLUSTERS_FILE = 'clusters_output.csv'
OUTPUT_ISOLATED_FILE = 'isolated_fes.csv'


# --- Data Loading and Preparation ---

def load_data(fe_filepath, relations_filepath):
    """
    Loads the Frame Element and relations data from CSV files, with detailed validation.
    """
    try:
        df_fe = pd.read_csv(fe_filepath)
        df_relations = pd.read_csv(relations_filepath)
        print("Data loaded successfully.")
        required_fe_cols = ['idFE', 'frameName', 'feName']
        missing_fe_cols = [col for col in required_fe_cols if col not in df_fe.columns]
        if missing_fe_cols:
            print(
                f"\n--- VALIDATION ERROR in '{fe_filepath}' ---\nMissing: {missing_fe_cols}\nAvailable: {list(df_fe.columns)}\n")
            return None, None
        required_relations_cols = ['superFE', 'subFE']
        missing_relations_cols = [col for col in required_relations_cols if col not in df_relations.columns]
        if missing_relations_cols:
            print(
                f"\n--- VALIDATION ERROR in '{relations_filepath}' ---\nMissing: {missing_relations_cols}\nAvailable: {list(df_relations.columns)}\n")
            return None, None
        print("Column validation passed.")
        return df_fe, df_relations
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the same directory.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None, None


def create_unique_fe_name(frame, fe):
    """Creates a standardized, unique name for a Frame Element node."""
    return f"{frame}.{fe}"


# --- Cluster and Network Construction ---

def build_clusters(df_fe, df_relations):
    """
    Builds clusters based on FE relations and returns the raw cluster data
    and a set of all FEs that were included in any cluster.
    """
    # Step 1: Create a map from FE ID to unique FE name.
    print("Step 1: Creating a map from FE ID to unique FE name...")
    id_to_fe_name_map = {row['idFE']: create_unique_fe_name(row['frameName'], row['feName']) for _, row in
                         df_fe.iterrows()}
    print(f"Map created with {len(id_to_fe_name_map)} entries.")

    # Step 2: Iteratively build clusters based on relations.
    print("Step 2: Building clusters by iterating through relations...")
    fe_to_hub = {}  # Maps an FE node to its hub name
    hub_to_fes = {}  # Maps a hub name to the set of FEs it contains
    hub_counter = 0

    relations = [
        (id_to_fe_name_map.get(row['superFE']), id_to_fe_name_map.get(row['subFE']))
        for _, row in df_relations.iterrows()
    ]

    for source_node, target_node in relations:
        if not source_node or not target_node:
            continue  # Skip relations with invalid IDs

        source_hub = fe_to_hub.get(source_node)
        target_hub = fe_to_hub.get(target_node)

        if source_hub is None and target_hub is None:
            # Case 1: Neither FE is in a cluster. Create a new one.
            hub_name = f"ClusterHub_{hub_counter}"
            hub_counter += 1
            hub_to_fes[hub_name] = {source_node, target_node}
            fe_to_hub[source_node] = hub_name
            fe_to_hub[target_node] = hub_name
        elif source_hub is not None and target_hub is None:
            # Case 2: Source is in a cluster, target is not. Add target to source's cluster.
            hub_to_fes[source_hub].add(target_node)
            fe_to_hub[target_node] = source_hub
        elif source_hub is None and target_hub is not None:
            # Case 3: Target is in a cluster, source is not. Add source to target's cluster.
            hub_to_fes[target_hub].add(source_node)
            fe_to_hub[source_node] = target_hub
        # elif source_hub != target_hub:
        #     # Case 4: Both are in different clusters. Merge them.
        #     if len(hub_to_fes[source_hub]) < len(hub_to_fes[target_hub]):
        #         source_hub, target_hub = target_hub, source_hub
        #     fes_to_move = hub_to_fes[target_hub]
        #     hub_to_fes[source_hub].update(fes_to_move)
        #     for fe in fes_to_move:
        #         fe_to_hub[fe] = source_hub
        #     del hub_to_fes[target_hub]

    print(f"Clustering complete. Found {len(hub_to_fes)} meaningful clusters.")

    # Return the clusters and the set of all FEs that are in any cluster
    all_clustered_fes = set(fe_to_hub.keys())
    return hub_to_fes, all_clustered_fes


# --- Data Export ---

def export_clusters_to_csv(clusters_data, filename):
    """
    Saves the hub and FE cluster data to a CSV file.
    """
    print(f"\nStep 3: Exporting cluster data to '{filename}'...")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['HubNode', 'FrameElements'])
            for hub_name, fe_set in clusters_data.items():
                fes_string = ";".join(sorted(list(fe_set)))
                writer.writerow([hub_name, fes_string])
        print("Export successful.")
    except IOError as e:
        print(f"Error writing to file: {e}")


def export_isolated_fes_to_csv(isolated_fes, filename):
    """
    Saves the list of isolated (non-clustered) FEs to a CSV file.
    """
    print(f"Step 4: Exporting isolated FE data to '{filename}'...")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['IsolatedFrameElement'])
            for fe_name in sorted(list(isolated_fes)):
                writer.writerow([fe_name])
        print("Export successful.")
    except IOError as e:
        print(f"Error writing to file: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    df_fe_all, df_fe_relations = load_data(FE_ALL_FILE, FE_RELATIONS_FILE)

    if df_fe_all is not None and df_fe_relations is not None:
        # Build the clusters and get the set of all FEs in those clusters
        final_clusters, clustered_fes = build_clusters(df_fe_all, df_fe_relations)

        # Export the cluster results to a CSV file for inspection
        export_clusters_to_csv(final_clusters, OUTPUT_CLUSTERS_FILE)

        # Determine the isolated FEs
        all_fes = {create_unique_fe_name(row['frameName'], row['feName']) for _, row in df_fe_all.iterrows()}
        isolated_fes = all_fes - clustered_fes
        print(f"\nFound {len(isolated_fes)} isolated Frame Elements.")

        # Export the isolated FEs to a second CSV file
        export_isolated_fes_to_csv(isolated_fes, OUTPUT_ISOLATED_FILE)
