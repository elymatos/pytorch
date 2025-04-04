from htpc_graph_builder import HTCPGraphBuilder

# Create a builder instance with debug mode enabled
builder = HTCPGraphBuilder(debug=True)

# Configure thresholds
builder.thresholds = {
    "bigram": 2,  # Minimum frequency to create a bigram
    "phrase": 2,  # Minimum frequency to create a phrase
    "hierarchy": 1,  # Minimum frequency to create a hierarchy
    "discourse": 1  # Minimum frequency to create a discourse pattern
}

# Build the graph from the example file
print("Building graph from test_input.txt...")
if builder.build_from_file("test_input.txt"):
    print("Graph built successfully.")

    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total nodes: {len(builder.graph['nodes'])}")
    print(f"AND nodes: {builder.graph['metadata']['and_count']}")
    print(f"OR nodes: {builder.graph['metadata']['or_count']}")
    print(f"L1 nodes: {builder.graph['metadata']['levels']['L1']}")
    print(f"L2 nodes: {builder.graph['metadata']['levels']['L2']}")
    print(f"L3 nodes: {builder.graph['metadata']['levels']['L3']}")
    print(f"L4 nodes: {builder.graph['metadata']['levels']['L4']}")
    print(f"L5 nodes: {builder.graph['metadata']['levels']['L5']}")
    print(f"Total edges: {len(builder.graph['edges'])}")

    # Save the graph to a JSON file
    if builder.save_to_json("htpc_model.json"):
        print("\nGraph saved to htpc_model.json")
    else:
        print("\nFailed to save graph.")

    # Generate visualization
    print("\nGenerating visualization...")
    try:
        if builder.visualize("htpc_graph.png", max_nodes=50, show=False):
            print("Visualization saved to htpc_graph.png")
        else:
            print("Failed to create visualization.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback

        traceback.print_exc()
else:
    print("Failed to build graph.")