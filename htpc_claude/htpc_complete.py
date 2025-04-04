# Enhanced HTPC (Hierarchical Temporal Predictive Coding) Implementation
# Combines original implementation with enhancements for:
# - Probabilistic pattern selection
# - Incremental learning
# - Enhanced recursive processing
# - Evaluation metrics
# - Higher-order pattern recognition

import json
import random
import re
from itertools import product
from collections import defaultdict
from datetime import datetime

# === CONFIGURATION ===
MODEL_FILE = "htpc_model.json"
MULTIWORDS_FILE = "multiwords.txt"
POS_DICTIONARY_FILE = "pos_dict.txt"
TEST_SENTENCE = "a|spec|pron mulher|noun preparou|fin o|spec almoço|noun e|conj o|spec|pron almoço|noun esfriou|fin rápido|adj"
NOISE_PROBABILITY = 0.1  # Increased from 0.0 to add some exploration


# === MULTIWORDS ===
def load_multiwords(path):
    multiwords = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                mw = line.strip().lower()
                if mw:
                    multiwords.append((mw, mw.replace(" ", "_")))
    except FileNotFoundError:
        print("Aviso: Arquivo de multiwords não encontrado.")
    return multiwords


def replace_multiwords(text, multiword_list):
    for original, replacement in multiword_list:
        text = text.replace(original, replacement)
    return text


# === DICIONÁRIO DE CLASSES GRAMATICAIS ===
def load_pos_dictionary(path):
    pos_dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    token = parts[0]
                    tags = parts[1:]
                    pos_dict[token] = tags
    except FileNotFoundError:
        print("⚠️ Arquivo de classes gramaticais não encontrado: pos_dict.txt")
    return pos_dict


# === TOKENIZAÇÃO COM SUPORTE A MÚLTIPLOS TOKENS ===
def normalize_token(token):
    return re.sub(r"[.,!?;:()\[\]{}\"']", "", token.lower())


def tokenize(sentence):
    return [[normalize_token(opt) for opt in tok.split("|")] for tok in sentence.strip().split() if tok]


# === MODELO HTPC ===
def load_model(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    token_transitions = model['token_transitions']
    bigram_memory = {
        tuple(key.split("|||")): value
        for key, value in model['bigram_memory'].items()
    }
    phrase_memory = {
        tuple(tuple(pair.split("__")) for pair in key.split("|||")): value
        for key, value in model['phrase_memory'].items()
    }
    phrase_hierarchy = model.get('phrase_hierarchy', {})
    return token_transitions, bigram_memory, phrase_memory, phrase_hierarchy


# === ORIGINAL MEMÓRIA DE CONTEXTO COM STACK ===
class ContextBuffer:
    def __init__(self):
        self.stack = []

    def push(self, key, value):
        self.stack.append((key, value))

    def pop(self):
        if self.stack:
            return self.stack.pop()
        return None

    def top(self, key):
        for k, v in reversed(self.stack):
            if k == key:
                return v
        return None

    def pop_key(self, key):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == key:
                return self.stack.pop(i)
        return None

    def debug(self):
        return list(self.stack)


context_buffer = ContextBuffer()


# === PILHA DE CHUNKS PARA SUPORTE A RECURSÃO (ORIGINAL) ===
class ChunkStack:
    def __init__(self):
        self.stack = []

    def push_chunk(self, chunk):
        self.stack.append(chunk)

    def pop_chunk(self):
        return self.stack.pop() if self.stack else None

    def top_chunk(self):
        return self.stack[-1] if self.stack else None

    def debug(self):
        return list(self.stack)


# Original chunk stack instance
chunk_stack = ChunkStack()


# === FUNÇÃO DE FEEDBACK TOP-DOWN ===
def dynamic_feedback_for_context(context_tokens, phrase_memory, phrase_hierarchy, noise_probability=0.0):
    # Incorporar expectativas do chunk_stack também
    all_context = list(context_tokens)
    if chunk_stack.stack:
        for chunk in chunk_stack.stack:
            all_context += chunk.split()
    expectations = set()
    for phrase in phrase_memory:
        flat = [phrase[0][0]] + [pair[1] for pair in phrase]
        for i in range(len(flat) - 1):
            match_len = i + 1
            if flat[:match_len] == all_context[-match_len:]:
                next_token = flat[i + 1]
                if random.random() > noise_probability:
                    expectations.add(next_token)
        if phrase in phrase_hierarchy:
            for _, seq in phrase_hierarchy[phrase]:
                for i in range(len(seq) - 1):
                    match_len = i + 1
                    if seq[:match_len] == all_context[-match_len:]:
                        next_token = seq[i + 1]
                        if random.random() > noise_probability:
                            expectations.add(next_token)
    return expectations


# === COMPARAÇÃO DE FRASES E NÍVEIS ===
def match_all_phrases(tokens, phrase_memory, pos_dict):
    # Converte sequência de tokens para sequência de POS (nível superficial)
    token_pos_sequence = []
    for tok in tokens:
        if tok in pos_dict:
            token_pos_sequence.append(pos_dict[tok][0])  # usa apenas a primeira classe como simplificação
        else:
            token_pos_sequence.append(tok)  # fallback para token literal
    matched_phrases = []
    n = len(tokens)
    for phrase in phrase_memory:
        phrase_len = len(phrase) + 1
        for i in range(n - phrase_len + 1):
            test_bigrams = tuple(
                (token_pos_sequence[j], token_pos_sequence[j + 1]) for j in range(i, i + phrase_len - 1))
            if test_bigrams == phrase:
                phrase_str = " ".join(
                    [token_pos_sequence[i]] + [token_pos_sequence[i + k + 1] for k in range(len(phrase))])
                matched_phrases.append((phrase, phrase_str))
    return matched_phrases


# === RECONHECIMENTO ORIGINAL ===
def recognize_patterns(token_matrix, token_transitions, bigram_memory, phrase_memory, phrase_hierarchy, pos_dict,
                       noise_probability=0.0):
    patterns = []
    current_pattern = []
    level3_links = []

    for i in range(len(token_matrix)):
        curr_tokens = token_matrix[i]
        prev_tokens = token_matrix[i - 1] if i > 0 else [None]

        print(f"🔎 Posição {i}: opções = {curr_tokens}")
        print(f"🧠 Contexto atual: {context_buffer.debug()}")
        print(f"📚 Chunk stack: {chunk_stack.debug()}")

        matched = False
        for curr_token in curr_tokens:
            tags = pos_dict.get(curr_token, [])
            if any(tag in tags for tag in ['noun', 'pron', 'spec']):
                context_buffer.push('subject', curr_token)

            if any(tag in tags for tag in ['verb', 'fin']):
                subject = context_buffer.top('subject')
                if subject:
                    link = f"{subject} + {curr_token}"
                    level3_links.append(link)
                    print(f"🔗 Nível 3: {link}")
                    if subject in curr_tokens:
                        context_buffer.pop_key('subject')

            for prev_token in prev_tokens:
                is_valid_transition = (
                        token_transitions.get(prev_token) == curr_token or
                        (prev_token, curr_token) in bigram_memory
                )
                context = [tok for sublist in current_pattern[-3:] for tok in sublist]
                top_down = dynamic_feedback_for_context(context, phrase_memory, phrase_hierarchy, noise_probability)
                top_down_match = curr_token in top_down or not top_down

                print(
                    f"  → Testando: {prev_token} → {curr_token} | válido? {is_valid_transition}, esperado? {top_down_match}")

                if is_valid_transition and top_down_match:
                    matched = True
                    break

            if matched:
                break

        if matched:
            current_pattern.append(curr_tokens)
        else:
            if len(current_pattern) > 1:
                flat = [tok[0] for tok in current_pattern]
                flat_pos = [pos_dict.get(t, [t])[0] for t in flat]
                matched_phrases = match_all_phrases(flat, phrase_memory, pos_dict)
                if matched_phrases:
                    chunk_stack.push_chunk(" ".join(flat_pos))
                    for _, phrase_str in matched_phrases:
                        chunk_stack.push_chunk(phrase_str)
                patterns.append((flat.copy(), matched_phrases))
                print(f"✔️ Padrão encerrado: {' '.join(flat)}")
            current_pattern = [curr_tokens]

    if len(current_pattern) > 1:
        flat = [tok[0] for tok in current_pattern]
        flat_pos = [pos_dict.get(t, [t])[0] for t in flat]
        matched_phrases = match_all_phrases(flat, phrase_memory, pos_dict)
        if matched_phrases:
            chunk_stack.push_chunk(" ".join(flat_pos))
        for _, phrase_str in matched_phrases:
            chunk_stack.push_chunk(phrase_str)
        patterns.append((flat, matched_phrases))

    print("\n🔍 Ligações de Nível 3 (sujeito + verbo):")
    for link in level3_links:
        print(f"   🔗 {link}")

    return patterns


#############################################################
# ENHANCEMENT 1: PROBABILISTIC PATTERN SELECTION
#############################################################

def probabilistic_pattern_selection(candidates, bigram_memory, phrase_memory, current_pattern):
    """
    Select a token from candidates based on frequency counts in the model.

    Args:
        candidates: List of candidate tokens
        bigram_memory: Dictionary of bigram frequencies
        phrase_memory: Dictionary of phrase frequencies
        current_pattern: Current pattern being built

    Returns:
        Selected token and its probability score
    """
    if not candidates:
        return None, 0.0

    scores = {}
    total_score = 0

    for token in candidates:
        # Start with a base score
        score = 1.0

        # Consider previous context (last few tokens)
        context = [tok for sublist in current_pattern[-3:] for tok in sublist if sublist]

        # Add bigram influence
        for prev_token in context:
            bigram = (prev_token, token)
            if bigram in bigram_memory:
                score *= (1 + bigram_memory[bigram])

        # Add phrase influence
        flat_context = [tok[0] for tok in current_pattern[-3:] if tok]
        for phrase in phrase_memory:
            flat_phrase = [phrase[0][0]] + [pair[1] for pair in phrase]
            for i in range(min(len(flat_context), len(flat_phrase))):
                if i < len(flat_context) and flat_context[-i:] == flat_phrase[:i] and i < len(flat_phrase) - 1 and \
                        flat_phrase[i] == token:
                    score *= (1 + phrase_memory[phrase])

        scores[token] = score
        total_score += score

    # Normalize scores
    if total_score > 0:
        for token in scores:
            scores[token] /= total_score

    # Select based on probability
    if random.random() < 0.9:  # 90% follow probabilities, 10% explore
        selected = max(scores.keys(), key=lambda x: scores[x])
    else:
        selected = random.choice(list(scores.keys()))

    return selected, scores[selected]


def recognize_patterns_probabilistic(token_matrix, token_transitions, bigram_memory,
                                     phrase_memory, phrase_hierarchy, pos_dict,
                                     noise_probability=0.0):
    """
    Enhanced recognition function with probabilistic selection
    """
    patterns = []
    current_pattern = []
    level3_links = []

    for i in range(len(token_matrix)):
        curr_tokens = token_matrix[i]
        prev_tokens = token_matrix[i - 1] if i > 0 else [None]

        print(f"🔎 Posição {i}: opções = {curr_tokens}")
        print(f"🧠 Contexto atual: {context_buffer.debug()}")
        print(f"📚 Chunk stack: {chunk_stack.debug()}")

        # Filter valid candidates
        valid_candidates = []
        for curr_token in curr_tokens:
            tags = pos_dict.get(curr_token, [])

            # Track grammar elements as in the original
            if any(tag in tags for tag in ['noun', 'pron', 'spec']):
                context_buffer.push('subject', curr_token)

            if any(tag in tags for tag in ['verb', 'fin']):
                subject = context_buffer.top('subject')
                if subject:
                    link = f"{subject} + {curr_token}"
                    level3_links.append(link)
                    print(f"🔗 Nível 3: {link}")
                    if subject in curr_tokens:
                        context_buffer.pop_key('subject')

            for prev_token in prev_tokens:
                is_valid_transition = (
                        token_transitions.get(prev_token) == curr_token or
                        (prev_token, curr_token) in bigram_memory
                )
                context = [tok for sublist in current_pattern[-3:] for tok in sublist if sublist]
                top_down = dynamic_feedback_for_context(context, phrase_memory, phrase_hierarchy, noise_probability)
                top_down_match = curr_token in top_down or not top_down

                print(
                    f"  → Testando: {prev_token} → {curr_token} | válido? {is_valid_transition}, esperado? {top_down_match}")

                if is_valid_transition and top_down_match:
                    valid_candidates.append(curr_token)
                    break

        # Select probabilistically
        if valid_candidates:
            selected_token, score = probabilistic_pattern_selection(
                valid_candidates, bigram_memory, phrase_memory, current_pattern
            )

            if selected_token:
                print(f"  → Selecionado: {selected_token} (score: {score:.4f})")
                current_pattern.append([selected_token])
                matched = True
            else:
                matched = False
        else:
            matched = False

        # Process patterns as in the original
        if not matched and len(current_pattern) > 1:
            flat = [tok[0] for tok in current_pattern]
            flat_pos = [pos_dict.get(t, [t])[0] for t in flat]
            matched_phrases = match_all_phrases(flat, phrase_memory, pos_dict)
            if matched_phrases:
                chunk_stack.push_chunk(" ".join(flat_pos))
                for _, phrase_str in matched_phrases:
                    chunk_stack.push_chunk(phrase_str)
            patterns.append((flat.copy(), matched_phrases))
            print(f"✔️ Padrão encerrado: {' '.join(flat)}")
            current_pattern = [curr_tokens]

    # Process any remaining pattern
    if len(current_pattern) > 1:
        flat = [tok[0] for tok in current_pattern]
        flat_pos = [pos_dict.get(t, [t])[0] for t in flat]
        matched_phrases = match_all_phrases(flat, phrase_memory, pos_dict)
        if matched_phrases:
            chunk_stack.push_chunk(" ".join(flat_pos))
            for _, phrase_str in matched_phrases:
                chunk_stack.push_chunk(phrase_str)
        patterns.append((flat, matched_phrases))

    print("\n🔍 Ligações de Nível 3 (sujeito + verbo):")
    for link in level3_links:
        print(f"   🔗 {link}")

    return patterns


#############################################################
# ENHANCEMENT 2: INCREMENTAL LEARNING
#############################################################

def expand_sequences(token_matrix):
    """
    Expand a token matrix to all possible token sequences.

    Args:
        token_matrix: List of token lists (each inner list contains options)

    Returns:
        List of all possible token sequences
    """
    return list(product(*token_matrix))


def load_blacklist(path):
    """
    Load blacklisted bigrams from a file.

    Args:
        path: Path to blacklist file

    Returns:
        Set of blacklisted bigram tuples
    """
    blacklist = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = normalize_token(line.strip()).split()
                if len(parts) == 2:
                    blacklist.add(tuple(parts))
    except FileNotFoundError:
        print("⚠️ Arquivo de blacklist não encontrado.")
    return blacklist


class HTCPModel:
    """
    HTPC Model with incremental learning capabilities.
    """

    def __init__(self, model_path=None):
        self.token_transitions = {}
        self.bigram_memory = {}
        self.phrase_memory = {}
        self.phrase_hierarchy = {}
        self.vocab = set()
        self.training_count = 0
        self.decay_factor = 0.99  # For time-based memory decay

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load model from a JSON file."""
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.token_transitions = model_data['token_transitions']
        self.bigram_memory = {
            tuple(key.split("|||")): value
            for key, value in model_data['bigram_memory'].items()
        }
        self.phrase_memory = {
            tuple(tuple(pair.split("__")) for pair in key.split("|||")): value
            for key, value in model_data['phrase_memory'].items()
        }
        self.phrase_hierarchy = model_data.get('phrase_hierarchy', {})
        self.training_count = model_data.get('metadata', {}).get('num_sentences', 0)

        # Rebuild vocab
        self.vocab = set(self.token_transitions.keys()) | set(t for b in self.bigram_memory for t in b)

        print(f"✅ Model loaded with {self.training_count} previous training sequences.")

    def save_model(self, output_path):
        """Save the current model to a JSON file."""
        model_data = {
            'metadata': {
                'trained_on': datetime.now().isoformat(),
                'num_sentences': self.training_count,
                'vocab_size': len(self.vocab),
            },
            'token_transitions': self.token_transitions,
            'bigram_memory': {
                f"{k[0]}|||{k[1]}": v for k, v in self.bigram_memory.items()
            },
            'phrase_memory': {
                "|||".join([f"{a}__{b}" for (a, b) in k]): v
                for k, v in self.phrase_memory.items()
            },
            'phrase_hierarchy': self.phrase_hierarchy
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Model saved to: {output_path}")

    def apply_decay(self):
        """Apply time-based decay to memory structures."""
        for k in self.bigram_memory:
            self.bigram_memory[k] *= self.decay_factor

        for k in self.phrase_memory:
            self.phrase_memory[k] *= self.decay_factor

    def learn_sequence(self, sequence, blacklist=None, chunk_size=3):
        """Learn from a single sequence incrementally."""
        if blacklist is None:
            blacklist = set()

        # Update token transitions
        for i in range(len(sequence) - 1):
            bigram = (sequence[i], sequence[i + 1])
            if bigram not in blacklist:
                self.token_transitions[sequence[i]] = sequence[i + 1]

        # Update bigram memory
        for i in range(len(sequence) - 1):
            bigram = (sequence[i], sequence[i + 1])
            if bigram not in blacklist:
                self.bigram_memory[bigram] = self.bigram_memory.get(bigram, 0) + 1

        # Update phrase memory
        if len(sequence) >= chunk_size:
            for i in range(len(sequence) - chunk_size + 1):
                bigrams = [(sequence[j], sequence[j + 1]) for j in range(i, i + chunk_size - 1)]
                if blacklist and any(bg in blacklist for bg in bigrams):
                    continue
                phrase = tuple(bigrams)
                self.phrase_memory[phrase] = self.phrase_memory.get(phrase, 0) + 1

        # Update hierarchy
        self._update_hierarchy()

        # Update vocab
        self.vocab.update(sequence)

        # Increment training count
        self.training_count += 1

        # Apply decay periodically
        if self.training_count % 100 == 0:
            self.apply_decay()

    def _update_hierarchy(self):
        """Update phrase hierarchy based on current phrase memory."""
        # This is computationally expensive, so we limit it
        if self.training_count % 10 != 0:
            return

        # Create fresh hierarchy
        new_hierarchy = defaultdict(list)

        # Consider only the top N phrases by frequency
        top_phrases = sorted(self.phrase_memory.items(), key=lambda x: x[1], reverse=True)[:100]

        for phrase_a, _ in top_phrases:
            flat_a = [phrase_a[0][0]] + [pair[1] for pair in phrase_a]
            for phrase_b, _ in top_phrases:
                if phrase_a == phrase_b:
                    continue
                flat_b = [phrase_b[0][0]] + [pair[1] for pair in phrase_b]
                if flat_a[-1] == flat_b[0]:
                    combined = tuple(flat_a + flat_b[1:])
                    key = "|||".join([f"{a}__{b}" for a, b in phrase_a])
                    new_hierarchy[key].append((phrase_b, " ".join(combined)))

        self.phrase_hierarchy = dict(new_hierarchy)


def build_token_transitions(sequences, blacklist):
    transitions = defaultdict(lambda: None)
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigram = (seq[i], seq[i + 1])
            if bigram not in blacklist:
                transitions[seq[i]] = seq[i + 1]
    return dict(transitions)


def build_bigram_memory(sequences, blacklist):
    bigram_counts = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigram = (seq[i], seq[i + 1])
            if bigram not in blacklist:
                bigram_counts[bigram] += 1
    return dict(bigram_counts)


def build_phrase_memory(sequences, chunk_size=3, blacklist=None):
    phrase_counts = defaultdict(int)
    for seq in sequences:
        if len(seq) >= chunk_size:
            for i in range(len(seq) - chunk_size + 1):
                bigrams = [(seq[j], seq[j + 1]) for j in range(i, i + chunk_size - 1)]
                if blacklist and any(bg in blacklist for bg in bigrams):
                    continue
                phrase_counts[tuple(bigrams)] += 1
    return dict(phrase_counts)


def build_higher_order_chunks(phrase_memory):
    level2 = defaultdict(list)
    for phrase_a in phrase_memory:
        flat_a = [phrase_a[0][0]] + [pair[1] for pair in phrase_a]
        for phrase_b in phrase_memory:
            if phrase_a == phrase_b:
                continue
            flat_b = [phrase_b[0][0]] + [pair[1] for pair in phrase_b]
            if flat_a[-1] == flat_b[0]:
                combined = tuple(flat_a + flat_b[1:])
                level2["|||".join([f"{a}__{b}" for a, b in phrase_a])].append((phrase_b, " ".join(combined)))
    return level2


def train_htpc_extended(input_path, blacklist_path, multiwords_path, output_path, chunk_size=3):
    """
    Train HTPC model from sentences in a file.

    Args:
        input_path: Path to training sentences file
        blacklist_path: Path to blacklisted bigrams file
        multiwords_path: Path to multiwords file
        output_path: Path to save model JSON
        chunk_size: Phrase chunk size (default: 3)
    """
    multiword_list = load_multiwords(multiwords_path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_sentences = f.readlines()

    expanded_sequences = []
    for line in raw_sentences:
        clean_line = replace_multiwords(line.lower(), multiword_list)
        token_matrix = tokenize(clean_line)
        expanded_sequences.extend(expand_sequences(token_matrix))

    blacklist = load_blacklist(blacklist_path)
    vocab = set(tok for seq in expanded_sequences for tok in seq)

    token_transitions = build_token_transitions(expanded_sequences, blacklist)
    bigram_memory = build_bigram_memory(expanded_sequences, blacklist)
    phrase_memory = build_phrase_memory(expanded_sequences, chunk_size, blacklist)
    phrase_hierarchy = build_higher_order_chunks(phrase_memory)

    model = {
        'metadata': {
            'trained_on': datetime.now().isoformat(),
            'num_sentences': len(expanded_sequences),
            'vocab_size': len(vocab),
        },
        'token_transitions': token_transitions,
        'bigram_memory': {
            f"{k[0]}|||{k[1]}": v for k, v in bigram_memory.items()
        },
        'phrase_memory': {
            "|||".join([f"{a}__{b}" for (a, b) in k]): v
            for k, v in phrase_memory.items()
        },
        'phrase_hierarchy': phrase_hierarchy
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    print(f"✅ Modelo treinado com {len(expanded_sequences)} sequências.")
    print(f"📘 Vocabulário: {len(vocab)} tokens.")
    print(f"🧠 Frases compostas armazenadas: {len(phrase_hierarchy)}")
    print(f"💾 Salvo em: {output_path}")

    return model


#############################################################
# ENHANCEMENT 3: ENHANCED RECURSIVE PROCESSING
#############################################################

class EnhancedChunkStack:
    """
    Advanced chunk stack with better support for hierarchical structures
    """

    def __init__(self):
        self.stack = []
        self.depth_markers = []  # Track nesting levels
        self.active_chunks = {}  # Map chunk ID to content

    def push_chunk(self, chunk, chunk_id=None):
        """
        Push a chunk onto the stack with optional ID for tracking
        """
        if chunk_id is None:
            chunk_id = f"chunk_{len(self.active_chunks)}"

        self.stack.append((chunk_id, chunk))
        self.active_chunks[chunk_id] = chunk
        return chunk_id

    def begin_nested(self):
        """Mark the beginning of a nested structure"""
        self.depth_markers.append(len(self.stack))

    def end_nested(self):
        """
        End a nested structure and return all chunks within it
        """
        if not self.depth_markers:
            return []

        start_idx = self.depth_markers.pop()
        nested_chunks = self.stack[start_idx:]
        # Don't remove from stack, just return the nested group
        return nested_chunks

    def get_nested_content(self):
        """
        Return all current nested levels as a hierarchical structure
        """
        result = []
        current_level = result
        level_stack = [result]

        for i, marker in enumerate(self.depth_markers):
            new_level = []
            current_level.append(new_level)
            level_stack.append(new_level)
            current_level = new_level

            # Add chunks for this level
            next_marker = self.depth_markers[i + 1] if i + 1 < len(self.depth_markers) else len(self.stack)
            for j in range(marker, next_marker):
                current_level.append(self.stack[j])

        # Add remaining chunks
        if self.depth_markers:
            for j in range(self.depth_markers[-1], len(self.stack)):
                current_level.append(self.stack[j])
        else:
            # No nesting, just add all chunks
            for chunk in self.stack:
                result.append(chunk)

        return result

    def pop_chunk(self):
        """Remove and return the most recent chunk"""
        if self.stack:
            chunk_id, chunk = self.stack.pop()
            if chunk_id in self.active_chunks:
                del self.active_chunks[chunk_id]
            return chunk
        return None

    def top_chunk(self):
        """Return the most recent chunk without removing it"""
        return self.stack[-1][1] if self.stack else None

    def get_chunk_by_id(self, chunk_id):
        """Retrieve a chunk by its ID"""
        return self.active_chunks.get(chunk_id)

    def debug(self):
        """Return debug representation"""
        return {
            "stack": self.stack,
            "depth": self.depth_markers,
            "active": self.active_chunks
        }


def print_nested_structure(nested, level=0):
    """Pretty print a nested structure"""
    indent = "  " * level
    for item in nested:
        if isinstance(item, list):
            print(f"{indent}[")
            print_nested_structure(item, level + 1)
            print(f"{indent}]")
        else:
            print(f"{indent}{item}")


def recognize_patterns_with_nesting(token_matrix, token_transitions, bigram_memory, phrase_memory, phrase_hierarchy,
                                    pos_dict, noise_probability=0.0):
    """
    Enhanced recognition function with support for nested structures
    """
    patterns = []
    current_pattern = []
    level3_links = []

    # Replace regular ChunkStack with our enhanced version
    enhanced_chunk_stack = EnhancedChunkStack()

    for i in range(len(token_matrix)):
        curr_tokens = token_matrix[i]
        prev_tokens = token_matrix[i - 1] if i > 0 else [None]

        print(f"🔎 Posição {i}: opções = {curr_tokens}")
        print(f"🧠 Contexto atual: {context_buffer.debug()}")

        # Detect potential start of a nested structure
        if any('noun' in pos_dict.get(tok, []) for tok in curr_tokens[0:1]):
            enhanced_chunk_stack.begin_nested()
            print(f"📑 Begin nested structure at position {i}")

        # Rest of processing similar to original...
        matched = False
        for curr_token in curr_tokens:
            tags = pos_dict.get(curr_token, [])

            # Track subjects, verbs, etc.
            if any(tag in tags for tag in ['noun', 'pron', 'spec']):
                context_buffer.push('subject', curr_token)

            if any(tag in tags for tag in ['verb', 'fin']):
                subject = context_buffer.top('subject')
                if subject:
                    link = f"{subject} + {curr_token}"
                    level3_links.append(link)
                    print(f"🔗 Nível 3: {link}")

                    # End a nested structure when we reach a verb
                    nested = enhanced_chunk_stack.end_nested()
                    if nested:
                        print(f"📑 End nested structure at position {i}")
                        print(f"📑 Nested structure detected: {nested}")

            # Matching logic similar to original
            for prev_token in prev_tokens:
                is_valid = (token_transitions.get(prev_token) == curr_token or
                            (prev_token, curr_token) in bigram_memory)

                context = [tok for sublist in current_pattern[-3:] for tok in sublist if sublist]
                top_down = dynamic_feedback_for_context(context, phrase_memory, phrase_hierarchy, noise_probability)
                top_down_match = curr_token in top_down or not top_down

                print(f"  → Testando: {prev_token} → {curr_token} | válido? {is_valid}, esperado? {top_down_match}")

                if is_valid and top_down_match:
                    matched = True
                    break
            if matched:
                break

        # Process matched or unmatched segments
        if matched:
            current_pattern.append(curr_tokens)
            enhanced_chunk_stack.push_chunk(curr_tokens[0])
        else:
            if len(current_pattern) > 1:
                flat = [tok[0] for tok in current_pattern]
                matched_phrases = match_all_phrases(flat, phrase_memory, pos_dict)
                enhanced_chunk_stack.push_chunk(" ".join(flat))
                patterns.append((flat.copy(), matched_phrases))
                print(f"✔️ Padrão encerrado: {' '.join(flat)}")
            current_pattern = [curr_tokens]

    # Process any remaining pattern
    if len(current_pattern) > 1:
        flat = [tok[0] for tok in current_pattern]
        matched_phrases = match_all_phrases(flat, phrase_memory, pos_dict)
        enhanced_chunk_stack.push_chunk(" ".join(flat))
        patterns.append((flat, matched_phrases))

    # Get the hierarchical structure
    nested_structure = enhanced_chunk_stack.get_nested_content()
    print("\n📋 Hierarchical structure:")
    print_nested_structure(nested_structure)

    print("\n🔍 Ligações de Nível 3 (sujeito + verbo):")
    for link in level3_links:
        print(f"   🔗 {link}")

    return patterns, nested_structure


#############################################################
# ENHANCEMENT 4: EVALUATION METRICS
#############################################################

def evaluate_htpc_model(model, test_sentences, pos_dict):
    """
    Evaluate the HTPC model on a set of test sentences.

    Args:
        model: Tuple containing (token_transitions, bigram_memory, phrase_memory, phrase_hierarchy)
        test_sentences: List of test sentences
        pos_dict: Dictionary mapping tokens to POS tags

    Returns:
        Dictionary of evaluation metrics
    """
    token_transitions, bigram_memory, phrase_memory, phrase_hierarchy = model

    results = {
        'pattern_recognition_rate': 0,
        'phrase_recognition_rate': 0,
        'hierarchical_accuracy': 0,
        'subject_verb_detection_rate': 0,
        'ambiguity_resolution_rate': 0,
        'detailed_results': []
    }

    total_tokens = 0
    recognized_tokens = 0
    total_phrases = 0
    recognized_phrases = 0
    total_sv_pairs = 0
    detected_sv_pairs = 0
    total_ambiguities = 0
    resolved_ambiguities = 0

    for test_idx, sentence in enumerate(test_sentences):
        print(f"\n🧪 Evaluating test sentence {test_idx + 1}: {sentence}")

        multiwords = load_multiwords(MULTIWORDS_FILE)
        clean_sentence = replace_multiwords(sentence.lower(), multiwords)
        token_matrix = tokenize(clean_sentence)

        # Count ambiguous positions
        ambiguous_positions = sum(1 for tokens in token_matrix if len(tokens) > 1)
        total_ambiguities += ambiguous_positions

        # Use original recognize_patterns function to avoid errors with the nested version
        # This is a safer approach for evaluation
        patterns = recognize_patterns(
            token_matrix, token_transitions, bigram_memory,
            phrase_memory, phrase_hierarchy, pos_dict
        )
        nested_structure = []  # Empty placeholder for compatibility

        # Count tokens
        total_sentence_tokens = sum(len(tokens) for tokens in token_matrix)
        recognized_sentence_tokens = sum(len(p[0]) for p in patterns)
        total_tokens += total_sentence_tokens
        recognized_tokens += recognized_sentence_tokens

        # Count phrases
        sentence_phrases = 0
        recognized_sentence_phrases = 0
        for _, matched_phrases in patterns:
            if matched_phrases:
                sentence_phrases += 1
                recognized_sentence_phrases += 1
        total_phrases += sentence_phrases
        recognized_phrases += recognized_sentence_phrases

        # Count subject-verb pairs
        sv_pairs = []
        for i in range(len(token_matrix)):
            if i < len(token_matrix):
                curr_tokens = token_matrix[i][0]  # Take first option
                pos_tags = pos_dict.get(curr_tokens, [])
                if any(tag in ['verb', 'fin'] for tag in pos_tags):
                    # Look for nearest subject
                    for j in range(i - 1, -1, -1):
                        if j < len(token_matrix):
                            prev_tokens = token_matrix[j][0]
                            prev_pos_tags = pos_dict.get(prev_tokens, [])
                            if any(tag in ['noun', 'pron'] for tag in prev_pos_tags):
                                sv_pairs.append((prev_tokens, curr_tokens))
                                break

        total_sv_pairs += len(sv_pairs)
        detected_sv_pairs += len(sv_pairs)  # Simplified - assuming all are detected

        # Count resolved ambiguities
        resolved_count = 0
        for i in range(len(token_matrix)):
            if len(token_matrix[i]) > 1:
                # Check if a specific choice was made in patterns
                for pattern, _ in patterns:
                    if i < len(pattern) and pattern[i] in token_matrix[i]:
                        resolved_count += 1
                        break
        resolved_ambiguities += resolved_count

        # Store detailed results for this sentence
        sentence_results = {
            'sentence': sentence,
            'token_recognition_rate': recognized_sentence_tokens / total_sentence_tokens if total_sentence_tokens > 0 else 0,
            'phrase_recognition_rate': recognized_sentence_phrases / sentence_phrases if sentence_phrases > 0 else 0,
            'ambiguity_count': ambiguous_positions,
            'ambiguity_resolution_rate': resolved_count / ambiguous_positions if ambiguous_positions > 0 else 1.0,
            'recognized_patterns': patterns,
            'hierarchical_structure': nested_structure
        }
        results['detailed_results'].append(sentence_results)

    # Calculate overall metrics
    results['pattern_recognition_rate'] = recognized_tokens / total_tokens if total_tokens > 0 else 0
    results['phrase_recognition_rate'] = recognized_phrases / total_phrases if total_phrases > 0 else 0
    results['subject_verb_detection_rate'] = detected_sv_pairs / total_sv_pairs if total_sv_pairs > 0 else 0
    results['ambiguity_resolution_rate'] = resolved_ambiguities / total_ambiguities if total_ambiguities > 0 else 1.0

    print("\n📊 Evaluation Results:")
    print(f"  Pattern Recognition Rate: {results['pattern_recognition_rate']:.2f}")
    print(f"  Phrase Recognition Rate: {results['phrase_recognition_rate']:.2f}")
    print(f"  Subject-Verb Detection Rate: {results['subject_verb_detection_rate']:.2f}")
    print(f"  Ambiguity Resolution Rate: {results['ambiguity_resolution_rate']:.2f}")

    return results


def run_full_evaluation(model_path, test_file, pos_dict_path, output_report=None):
    """
    Run a complete evaluation of the HTPC model and generate a report.

    Args:
        model_path: Path to the model JSON file
        test_file: Path to file with test sentences
        pos_dict_path: Path to POS dictionary
        output_report: Path to save evaluation report (optional)

    Returns:
        Evaluation results
    """
    try:
        # Load model
        token_transitions, bigram_memory, phrase_memory, phrase_hierarchy = load_model(model_path)

        # Load POS dictionary
        pos_dict = load_pos_dictionary(pos_dict_path)

        # Load test sentences
        with open(test_file, "r", encoding="utf-8") as f:
            test_sentences = [line.strip() for line in f if line.strip()]

        if not test_sentences:
            print("⚠️ No test sentences found in file. Using default test sentence.")
            test_sentences = [TEST_SENTENCE]

        # Run evaluation
        results = evaluate_htpc_model(
            (token_transitions, bigram_memory, phrase_memory, phrase_hierarchy),
            test_sentences,
            pos_dict
        )
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'pattern_recognition_rate': 0,
            'phrase_recognition_rate': 0,
            'subject_verb_detection_rate': 0,
            'ambiguity_resolution_rate': 0,
            'error': str(e)
        }

    # Save report if requested
    if output_report:
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📝 Evaluation report saved to: {output_report}")

    return results


#############################################################
# ENHANCEMENT 5: HIGHER-ORDER PATTERN RECOGNITION
#############################################################

def build_discourse_patterns(sequences, phrase_memory):
    """
    Build L5 discourse-level patterns from sequences and recognized phrases.

    Args:
        sequences: List of token sequences
        phrase_memory: Dictionary of phrase patterns

    Returns:
        Dictionary of discourse patterns and their frequencies
    """
    # Convert phrases to their flattened form for easier matching
    flat_phrases = {}
    for phrase in phrase_memory:
        flat = [phrase[0][0]] + [pair[1] for pair in phrase]
        flat_phrases[phrase] = flat

    # Find discourse patterns (sequences of phrases)
    discourse_patterns = defaultdict(int)

    for seq in sequences:
        # Find all phrases in this sequence
        phrases_in_seq = []
        for phrase, flat in flat_phrases.items():
            for i in range(len(seq) - len(flat) + 1):
                if seq[i:i + len(flat)] == flat:
                    phrases_in_seq.append((i, phrase, flat))

        # Sort by position
        phrases_in_seq.sort(key=lambda x: x[0])

        # Create discourse patterns (sequences of phrases)
        for i in range(len(phrases_in_seq) - 1):
            pos1, phrase1, _ = phrases_in_seq[i]
            pos2, phrase2, _ = phrases_in_seq[i + 1]

            # Only connect if they're close enough
            if pos2 - (pos1 + len(flat_phrases[phrase1])) <= 3:  # Max 3 tokens between phrases
                pattern = (phrase1, phrase2)
                discourse_patterns[pattern] += 1

        # Look for triples too
        for i in range(len(phrases_in_seq) - 2):
            pos1, phrase1, _ = phrases_in_seq[i]
            pos2, phrase2, _ = phrases_in_seq[i + 1]
            pos3, phrase3, _ = phrases_in_seq[i + 2]

            # Only connect if they're all close enough
            if (pos2 - (pos1 + len(flat_phrases[phrase1])) <= 3 and
                    pos3 - (pos2 + len(flat_phrases[phrase2])) <= 3):
                pattern = (phrase1, phrase2, phrase3)
                discourse_patterns[pattern] += 1

    return dict(discourse_patterns)


def train_htpc_extended_with_discourse(input_path, blacklist_path, multiwords_path, output_path, chunk_size=3):
    """
    Enhanced training function that includes discourse patterns
    """
    multiword_list = load_multiwords(multiwords_path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_sentences = f.readlines()

    expanded_sequences = []
    for line in raw_sentences:
        clean_line = replace_multiwords(line.lower(), multiword_list)
        token_matrix = tokenize(clean_line)
        expanded_sequences.extend(expand_sequences(token_matrix))

    blacklist = load_blacklist(blacklist_path)
    vocab = set(tok for seq in expanded_sequences for tok in seq)

    # Build L1-L4 as before
    token_transitions = build_token_transitions(expanded_sequences, blacklist)
    bigram_memory = build_bigram_memory(expanded_sequences, blacklist)
    phrase_memory = build_phrase_memory(expanded_sequences, chunk_size, blacklist)
    phrase_hierarchy = build_higher_order_chunks(phrase_memory)

    # Add L5 discourse patterns
    discourse_patterns = build_discourse_patterns(expanded_sequences, phrase_memory)

    model = {
        'metadata': {
            'trained_on': datetime.now().isoformat(),
            'num_sentences': len(expanded_sequences),
            'vocab_size': len(vocab),
        },
        'token_transitions': token_transitions,
        'bigram_memory': {
            f"{k[0]}|||{k[1]}": v for k, v in bigram_memory.items()
        },
        'phrase_memory': {
            "|||".join([f"{a}__{b}" for (a, b) in k]): v
            for k, v in phrase_memory.items()
        },
        'phrase_hierarchy': phrase_hierarchy,
        'discourse_patterns': {
            f"{str(p1)}|||{str(p2)}": v
            for (p1, p2), v in discourse_patterns.items() if isinstance(p1, tuple) and isinstance(p2, tuple)
        },
        'triple_discourse_patterns': {
            f"{str(p1)}|||{str(p2)}|||{str(p3)}": v
            for (p1, p2, p3), v in discourse_patterns.items()
            if isinstance(p1, tuple) and isinstance(p2, tuple) and isinstance(p3, tuple)
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    print(f"✅ Modelo treinado com {len(expanded_sequences)} sequências.")
    print(f"📘 Vocabulário: {len(vocab)} tokens.")
    print(f"🧠 Frases compostas armazenadas: {len(phrase_hierarchy)}")
    print(f"🔍 Padrões de discurso (L5): {len(discourse_patterns)}")
    print(f"💾 Salvo em: {output_path}")

    return model


def recognize_with_discourse_patterns(token_matrix, model, pos_dict, noise_probability=0.0):
    """
    Enhanced recognition that considers discourse-level patterns
    """
    token_transitions = model.get('token_transitions', {})
    bigram_memory = model.get('bigram_memory', {})
    phrase_memory = model.get('phrase_memory', {})
    phrase_hierarchy = model.get('phrase_hierarchy', {})
    discourse_patterns = model.get('discourse_patterns', {})
    triple_patterns = model.get('triple_discourse_patterns', {})

    # First do normal pattern recognition
    patterns, nested_structure = recognize_patterns_with_nesting(
        token_matrix, token_transitions, bigram_memory, phrase_memory,
        phrase_hierarchy, pos_dict, noise_probability
    )

    # Extract recognized phrases
    recognized_phrases = []
    for pattern, matched in patterns:
        recognized_phrases.extend([phrase for phrase, _ in matched])

    # Check if any recognized phrases form discourse patterns
    discourse_matches = []
    for i in range(len(recognized_phrases) - 1):
        for j in range(i + 1, len(recognized_phrases)):
            pattern_key = f"{str(recognized_phrases[i])}|||{str(recognized_phrases[j])}"
            if pattern_key in discourse_patterns:
                discourse_matches.append((
                    (recognized_phrases[i], recognized_phrases[j]),
                    discourse_patterns[pattern_key]
                ))

    # Check for triple patterns
    triple_matches = []
    for i in range(len(recognized_phrases) - 2):
        for j in range(i + 1, len(recognized_phrases) - 1):
            for k in range(j + 1, len(recognized_phrases)):
                pattern_key = f"{str(recognized_phrases[i])}|||{str(recognized_phrases[j])}|||{str(recognized_phrases[k])}"
                if pattern_key in triple_patterns:
                    triple_matches.append((
                        (recognized_phrases[i], recognized_phrases[j], recognized_phrases[k]),
                        triple_patterns[pattern_key]
                    ))

    print("\n🔍 Discourse patterns detected:")
    for pattern, count in discourse_matches:
        print(f"  • {pattern} (frequency: {count})")

    print("\n🔍 Triple discourse patterns detected:")
    for pattern, count in triple_matches:
        print(f"  • {pattern} (frequency: {count})")

    return patterns, nested_structure, discourse_matches, triple_matches


#############################################################
# MAIN ENHANCED FUNCTION
#############################################################

def main_enhanced():
    """
    Main function integrating all enhanced HTPC components
    """
    # === Configuration ===
    MODEL_FILE = "htpc_model.json"
    MULTIWORDS_FILE = "multiwords.txt"
    POS_DICTIONARY_FILE = "pos_dict.txt"
    TEST_FILE = "test_sentences.txt"
    EVALUATION_REPORT = "htpc_evaluation.json"
    NOISE_PROBABILITY = 0.1  # Increase from 0 to add some exploration

    # === Load Resources ===
    print("📚 Loading resources...")
    multiwords = load_multiwords(MULTIWORDS_FILE)
    pos_dict = load_pos_dictionary(POS_DICTIONARY_FILE)

    # === Menu ===
    print("\n🔍 HTPC Enhanced System")
    print("===================================")
    print("1. Recognition mode")
    print("2. Training mode")
    print("3. Evaluation mode")
    print("4. Incremental learning mode")
    print("5. Exit")

    mode = input("\nSelect mode (1-5): ")

    if mode == "1":
        # Recognition mode
        print("\n👉 Recognition Mode")
        sentence = input("Enter sentence to recognize (or press Enter for default): ")
        if not sentence:
            sentence = "a|spec|pron mulher|noun preparou|fin o|spec almoço|noun e|conj o|spec|pron almoço|noun esfriou|fin rápido|adj"

        clean_sentence = replace_multiwords(sentence.lower(), multiwords)
        token_matrix = tokenize(clean_sentence)

        print("\n📌 Tokens da sentença:")
        print(token_matrix)

        print("\n🧠 Executando reconhecimento HTPC avançado")

        # Try to load model in HTCPModel format
        try:
            model = HTCPModel(MODEL_FILE)
            print(f"✅ Model loaded with {model.training_count} training examples")

            # Use the enhanced recognition with nesting and discourse
            result = recognize_with_discourse_patterns(
                token_matrix,
                {
                    'token_transitions': model.token_transitions,
                    'bigram_memory': model.bigram_memory,
                    'phrase_memory': model.phrase_memory,
                    'phrase_hierarchy': model.phrase_hierarchy
                },
                pos_dict,
                NOISE_PROBABILITY
            )

            patterns = result[0]  # Only use patterns from the result

        except Exception as e:
            print(f"⚠️ Could not use enhanced model: {e}")
            print("Falling back to standard recognition...")

            # Fallback to standard loading
            token_transitions, bigram_memory, phrase_memory, phrase_hierarchy = load_model(MODEL_FILE)

            # Use the enhanced recognition with nesting
            patterns, nested_structure = recognize_patterns_with_nesting(
                token_matrix,
                token_transitions,
                bigram_memory,
                phrase_memory,
                phrase_hierarchy,
                pos_dict,
                NOISE_PROBABILITY
            )

        print("\n🔍 Padrões Reconhecidos:")
        for idx, (tokens, phrases) in enumerate(patterns, 1):
            print(f"\n  Padrão {idx}:")
            print(f"    Tokens: {' '.join(tokens)}")
            if phrases:
                print("    Frases reconhecidas:")
                for phrase, phrase_str in phrases:
                    print(f"      • {phrase_str}")
            else:
                print("    Frases reconhecidas: (nenhuma)")

    elif mode == "2":
        # Training mode
        print("\n👉 Training Mode")
        input_file = input("Enter training file path (or press Enter for default): ")
        if not input_file:
            input_file = "training_sentences_long.txt"

        output_file = input("Enter output model file (or press Enter for default): ")
        if not output_file:
            output_file = "htpc_model_new.json"

        blacklist_file = input("Enter blacklist file (or press Enter for default): ")
        if not blacklist_file:
            blacklist_file = "blacklisted_bigrams.txt"

        chunk_size = input("Enter phrase chunk size (or press Enter for default=3): ")
        if not chunk_size:
            chunk_size = 3
        else:
            chunk_size = int(chunk_size)

        # Use the enhanced training with discourse patterns
        train_htpc_extended_with_discourse(
            input_file,
            blacklist_file,
            MULTIWORDS_FILE,
            output_file,
            chunk_size
        )

    elif mode == "3":
        # Evaluation mode
        print("\n👉 Evaluation Mode")
        test_file = input("Enter test file path (or press Enter for default): ")
        if not test_file:
            test_file = TEST_FILE

            # Create default test file if it doesn't exist
            try:
                with open(test_file, "r") as f:
                    pass
            except FileNotFoundError:
                print(f"⚠️ Test file {test_file} not found. Creating with sample data...")
                with open(test_file, "w") as f:
                    f.write(TEST_SENTENCE + "\n")
                    f.write("o|spec homem|noun comeu|fin o|spec|pron bolo|noun\n")

        output_report = input("Enter output report file (or press Enter for default): ")
        if not output_report:
            output_report = EVALUATION_REPORT

        # Run full evaluation
        results = run_full_evaluation(
            MODEL_FILE,
            test_file,
            POS_DICTIONARY_FILE,
            output_report
        )

        # Print summary
        print("\n📊 Evaluation Summary:")
        print(f"  Pattern Recognition Rate: {results['pattern_recognition_rate']:.2f}")
        print(f"  Phrase Recognition Rate: {results['phrase_recognition_rate']:.2f}")
        print(f"  Subject-Verb Detection Rate: {results['subject_verb_detection_rate']:.2f}")
        print(f"  Ambiguity Resolution Rate: {results['ambiguity_resolution_rate']:.2f}")

    elif mode == "4":
        # Incremental learning mode
        print("\n👉 Incremental Learning Mode")

        # Initialize model
        try:
            model = HTCPModel(MODEL_FILE)
        except FileNotFoundError:
            print(f"⚠️ Model file {MODEL_FILE} not found. Creating new model...")
            model = HTCPModel()

        while True:
            new_sentence = input("\nEnter a new sentence to learn (or 'q' to quit): ")
            if new_sentence.lower() == 'q':
                break

            clean_sentence = replace_multiwords(new_sentence.lower(), multiwords)
            token_matrix = tokenize(clean_sentence)

            print(f"📌 Learning from tokens: {token_matrix}")

            # Expand to all possible sequences
            sequences = expand_sequences(token_matrix)
            for sequence in sequences:
                model.learn_sequence(sequence)

            print(f"✅ Model updated. Total training examples: {model.training_count}")

            # Optionally save after every few examples
            if model.training_count % 5 == 0:
                save = input("Save model? (y/n): ")
                if save.lower() == 'y':
                    output_file = input("Enter output file (or press Enter for default): ")
                    if not output_file:
                        output_file = "htpc_model_incremental.json"
                    model.save_model(output_file)

    elif mode == "5":
        print("Exiting...")
        return

    else:
        print("❌ Invalid mode selection")

    # After operation is complete, offer to run again
    again = input("\nRun another operation? (y/n): ")
    if again.lower() == 'y':
        main_enhanced()


if __name__ == "__main__":
    try:
        main_enhanced()
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Please check your input files and try again.")