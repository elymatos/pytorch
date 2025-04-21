import numpy as np
from sksequitur import parse
from collections import defaultdict


class RelationalNetworkNode:
    """
    A node in the Relational Network representing either a terminal symbol
    or a rule in the grammar
    """

    def __init__(self, label, is_terminal=False):
        self.label = label
        self.is_terminal = is_terminal

        # Ordered AND connections (for sequences)
        self.and_connections = []

        # OR connections (for alternatives)
        self.or_connections = []

        # Connection strengths (weights)
        self.connection_strengths = {}

        # Activation level
        self.activation = 0.0

        # Threshold for activation
        self.threshold = 0.5

    def add_and_connection(self, node, position, strength=1.0):
        """Add a connection in a sequence (ordered AND)"""
        while len(self.and_connections) <= position:
            self.and_connections.append(None)

        self.and_connections[position] = node
        self.connection_strengths[node.label] = strength

    def add_or_connection(self, node, strength=1.0):
        """Add an alternative connection (OR)"""
        if node not in self.or_connections:
            self.or_connections.append(node)
            self.connection_strengths[node.label] = strength

    def activate(self, value=1.0):
        """Activate this node"""
        self.activation = value
        return self.activation

    def propagate_down(self, depth=0):
        """Propagate activation down through AND connections"""
        if self.activation < self.threshold:
            return []

        # If this is a terminal, just return itself
        if self.is_terminal:
            return [self.label]

        # Propagate through AND connections (sequence)
        result = []
        for conn in self.and_connections:
            if conn is not None:
                # Apply activation with connection strength
                activation = self.activation * self.connection_strengths.get(conn.label, 1.0)
                conn.activate(activation)
                result.extend(conn.propagate_down(depth + 1))

        return result

    def propagate_up(self, input_sequence, position, depth=0):
        """Propagate activation upward from input"""
        # Base case: if we're at the end of input or too deep in recursion
        if position >= len(input_sequence) or depth > 10:
            return 0.0, 0

        # Terminal nodes directly match input
        if self.is_terminal:
            if position < len(input_sequence) and self.label == input_sequence[position]:
                return 1.0, 1  # Return activation and units consumed
            return 0.0, 0

        # For non-terminal nodes, try to match the full sequence of children
        total_activation = 0.0
        total_consumed = 0
        curr_pos = position

        for i, conn in enumerate(self.and_connections):
            if conn is None:
                continue

            # Get activation from this child
            activation, consumed = conn.propagate_up(input_sequence, curr_pos, depth + 1)

            # If this child doesn't match, the whole sequence fails
            if activation == 0.0:
                return 0.0, 0

            # Update position and totals
            curr_pos += consumed
            total_activation += activation
            total_consumed += consumed

        # Return the average activation and total consumed
        if len(self.and_connections) > 0:
            avg_activation = total_activation / len(self.and_connections)
            return avg_activation, total_consumed
        return 0.0, 0

    def __str__(self):
        if self.is_terminal:
            return f"Terminal({self.label})"
        return f"Rule({self.label})"


class RelationalNetwork:
    """
    A Relational Network built from a SEQUITUR grammar
    """

    def __init__(self):
        self.nodes = {}
        self.root_node = None

    def get_or_create_node(self, label, is_terminal=False):
        """Get an existing node or create a new one"""
        if label not in self.nodes:
            self.nodes[label] = RelationalNetworkNode(label, is_terminal)
        return self.nodes[label]

    def build_from_grammar_string(self, grammar_str):
        """
        Build a Relational Network from the string representation of a grammar
        """
        lines = grammar_str.strip().split('\n')
        rules = {}

        # First pass: parse all rules
        for line in lines:
            if ' -> ' not in line and ' → ' not in line:
                continue

            # Handle different arrow formats
            if ' -> ' in line:
                parts = line.split(' -> ')
            else:
                parts = line.split(' → ')

            if len(parts) != 2:
                continue

            rule_id = parts[0].strip()
            expansion = parts[1].strip().split(' ')
            rules[rule_id] = expansion

        # Second pass: create nodes and connections
        for rule_id, expansion in rules.items():
            # Create rule node
            rule_node = self.get_or_create_node(rule_id)

            # Create connections for each symbol in the expansion
            for i, symbol in enumerate(expansion):
                if len(symbol) == 0:
                    continue

                # Check if this is a rule reference or terminal
                if symbol in rules:
                    # Rule reference
                    symbol_node = self.get_or_create_node(symbol)
                    is_terminal = False
                else:
                    # Terminal symbol (single character or special symbol)
                    symbol_node = self.get_or_create_node(symbol, is_terminal=True)

                # Add connection with strength 1.0
                rule_node.add_and_connection(symbol_node, i, 1.0)

        # Set the root node (rule 0 or S)
        self.root_node = self.nodes.get('0')
        if not self.root_node:
            self.root_node = self.nodes.get('S')

    def recognize(self, input_sequence):
        """Recognize an input sequence"""
        if not self.root_node:
            return 0.0

        # Convert string to list of characters if needed
        if isinstance(input_sequence, str):
            input_sequence = list(input_sequence)

        # Reset all activations
        self.reset_activations()

        # Propagate activation upward
        activation, consumed = self.root_node.propagate_up(input_sequence, 0)

        # Calculate match score
        if consumed == len(input_sequence):
            return activation
        else:
            # Partial match - scale by proportion matched
            return activation * (consumed / len(input_sequence))

    def generate(self):
        """Generate a sequence from the network"""
        if not self.root_node:
            return []

        # Reset all activations
        self.reset_activations()

        # Activate the root node
        self.root_node.activate(1.0)

        # Propagate activation downward
        return self.root_node.propagate_down()

    def reset_activations(self):
        """Reset all node activations"""
        for node in self.nodes.values():
            node.activation = 0.0

    def print_structure(self):
        """Print the structure of the network"""
        for label, node in sorted(self.nodes.items()):
            connections = []
            for i, conn in enumerate(node.and_connections):
                if conn:
                    connections.append(f"{i}:{conn.label}")

            if node.is_terminal:
                node_type = "Terminal"
            else:
                node_type = "Rule"

            print(f"{node_type} {label}: AND connections: {', '.join(connections)}")


def sequitur_to_relational_network(input_sequence):
    """
    Process a sequence with SEQUITUR and build a Relational Network
    """
    # Parse the sequence with scikit-sequitur
    grammar = parse(input_sequence)
    grammar_str = str(grammar)

    # Build the Relational Network from the grammar string
    network = RelationalNetwork()
    network.build_from_grammar_string(grammar_str)

    return network, grammar_str


# Example usage
def main():
    # Example 1: Simple repeated sequence
    sequence1 = "abcabcabc"
    network1, grammar_str1 = sequitur_to_relational_network(sequence1)

    print("Example 1: abcabcabc")
    print("Grammar:")
    print(grammar_str1)

    print("\nRelational Network Structure:")
    network1.print_structure()

    print("\nRecognition tests:")
    print(f"Original sequence: {network1.recognize(sequence1)}")
    print(f"Modified sequence: {network1.recognize('abcabc')}")
    print(f"Different sequence: {network1.recognize('abcdabc')}")

    print("\nGeneration:")
    generated1 = ''.join(network1.generate())
    print(f"Generated sequence: {generated1}")

    # Example 2: More complex sequence with a pattern
    sequence2 = "abcdbc abcdbc"
    network2, grammar_str2 = sequitur_to_relational_network(sequence2)

    print("\n\nExample 2: abcdbc abcdbc")
    print("Grammar:")
    print(grammar_str2)

    print("\nRelational Network Structure:")
    network2.print_structure()

    print("\nRecognition tests:")
    print(f"Original sequence: {network2.recognize(sequence2)}")
    print(f"Partial sequence: {network2.recognize('abcdbc')}")

    print("\nGeneration:")
    generated2 = ''.join(network2.generate())
    print(f"Generated sequence: {generated2}")


if __name__ == "__main__":
    main()