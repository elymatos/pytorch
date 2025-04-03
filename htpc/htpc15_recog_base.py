import json
import random
import re
from itertools import product

# === CONFIGURAÇÃO ===
MODEL_FILE = "htpc_model.json"
MULTIWORDS_FILE = "multiwords.txt"
TEST_SENTENCE = "o homem preparou o café da manhã com cuidado"
NOISE_PROBABILITY = 0.0

# === MULTIWORDS ===
def load_multiwords(path):
    multiwords = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mw = line.strip().lower()
            if mw:
                multiwords.append((mw, mw.replace(" ", "_")))
    return multiwords

def replace_multiwords(text, multiword_list):
    for original, replacement in multiword_list:
        text = text.replace(original, replacement)
    return text

# === TOKENIZAÇÃO ===
def normalize_token(token):
    return re.sub(r"[.,!?;:()\\[\\]{}\\\"']", "", token.lower())

def tokenize(sentence):
    return [normalize_token(tok) for tok in sentence.strip().split() if tok]

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

# === MEMÓRIA DE CONTEXTO ===
class ContextBuffer:
    def __init__(self):
        self.buffer = {}

    def add(self, key, value):
        """Add a token to the buffer."""
        self.buffer[key] = value

    def get(self, key):
        """Retrieve a token from the buffer."""
        return self.buffer.get(key, None)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

# Instanciar a memória
context_buffer = ContextBuffer()

# === REMAPEAMENTO TOP-DOWN ===
def dynamic_feedback_for_context(context_tokens, phrase_memory, phrase_hierarchy, noise_probability=0.0):
    expectations = set()
    for phrase in phrase_memory:
        flat = [phrase[0][0]] + [pair[1] for pair in phrase]
        for i in range(len(flat) - 1):
            match_len = i + 1
            if flat[:match_len] == context_tokens[-match_len:]:
                next_token = flat[i + 1]
                if random.random() > noise_probability:
                    expectations.add(next_token)

        # nível 2
        if phrase in phrase_hierarchy:
            for _, seq in phrase_hierarchy[phrase]:
                for i in range(len(seq) - 1):
                    match_len = i + 1
                    if seq[:match_len] == context_tokens[-match_len:]:
                        next_token = seq[i + 1]
                        if random.random() > noise_probability:
                            expectations.add(next_token)
    return expectations

# === COMPARA FRASES ===
def match_all_phrases(tokens, phrase_memory):
    matched_phrases = []
    n = len(tokens)
    for phrase in phrase_memory:
        phrase_len = len(phrase) + 1
        for i in range(n - phrase_len + 1):
            test_bigrams = tuple((tokens[j], tokens[j + 1]) for j in range(i, i + phrase_len - 1))
            if test_bigrams == phrase:
                phrase_str = " ".join([tokens[i]] + [tokens[i + k + 1] for k in range(len(phrase))])
                matched_phrases.append(phrase_str)
    return matched_phrases

# === RECONHECIMENTO ===
def recognize_patterns(tokens, token_transitions, bigram_memory, phrase_memory, phrase_hierarchy, noise_probability=0.0):
    patterns = []
    current_pattern = []

    for i in range(len(tokens)):
        curr_token = tokens[i]
        prev_token = tokens[i - 1] if i > 0 else None

        # Armazenar sujeitos no buffer
        if curr_token in ['I', 'he', 'she', 'they']:  # Exemplo simplificado de sujeitos
            context_buffer.add('subject', curr_token)

        # Exemplo de verbos
        if curr_token in ['have', 'is', 'am']:  # Simplificação para verbos
            subject = context_buffer.get('subject')
            if subject:
                print(f"🔗 Ligando {subject} com o verbo {curr_token}")

        is_valid_transition = (
            token_transitions.get(prev_token) == curr_token or
            (prev_token, curr_token) in bigram_memory
        )

        context = current_pattern[-3:]
        dynamic_expectations = dynamic_feedback_for_context(context, phrase_memory, phrase_hierarchy, noise_probability)
        top_down_match = curr_token in dynamic_expectations

        print(f"🔎 Token atual: '{curr_token}' | Precedente: '{prev_token}'")
        print(f"   → Transição válida? {is_valid_transition}")
        print(f"   → Expectativas top-down: {dynamic_expectations} → Compatível? {top_down_match}")

        if is_valid_transition and top_down_match:
            current_pattern.append(curr_token)
        else:
            print(f"\n⚠️  Inibição em '{curr_token}' (posição {i})")
            if len(current_pattern) > 1:
                matched_phrases = match_all_phrases(current_pattern, phrase_memory)
                print(f"   ✔️ Frase encerrada: {' '.join(current_pattern)}")
                patterns.append((current_pattern.copy(), matched_phrases))
            current_pattern = [curr_token]

    if len(current_pattern) > 1:
        matched_phrases = match_all_phrases(current_pattern, phrase_memory)
        patterns.append((current_pattern, matched_phrases))

    return patterns

# === EXECUÇÃO ===
if __name__ == "__main__":
    random.seed(42)
    multiwords = load_multiwords(MULTIWORDS_FILE)
    sentence = replace_multiwords(TEST_SENTENCE.lower(), multiwords)
    tokens = tokenize(sentence)

    print("\n📌 Tokens gerados da sentença de teste:")
    print(tokens)

    token_transitions, bigram_memory, phrase_memory, phrase_hierarchy = load_model(MODEL_FILE)

    print("\n🧠 Reconhecimento HTPC com remapeamento dinâmico (ruído={NOISE_PROBABILITY})")
    patterns = recognize_patterns(tokens, token_transitions, bigram_memory, phrase_memory, phrase_hierarchy, NOISE_PROBABILITY)

    print("\n🔍 Padrões Reconhecidos:")
    for idx, (tokens, phrases) in enumerate(patterns, 1):
        print(f"\n  Padrão {idx}:")
        print(f"    Tokens: {' '.join(tokens)}")
        if phrases:
            print("    Frases correspondentes:")
            for p in phrases:
                print(f"      • {p}")
        else:
            print("    Frases correspondentes: (nenhuma)")
