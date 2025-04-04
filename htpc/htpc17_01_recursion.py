import json
import random
import re
from itertools import product

# === CONFIGURAÇÃO ===
MODEL_FILE = "htpc_model.json"
MULTIWORDS_FILE = "multiwords.txt"
POS_DICTIONARY_FILE = "pos_dict.txt"
TEST_SENTENCE = "a|spec|pron mulher|noun preparou|fin o|spec almoço|noun e|conj o|spec|pron almoço|noun esfriou|fin rápido|adj"
NOISE_PROBABILITY = 0.0

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

# === MEMÓRIA DE CONTEXTO COM STACK ===
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

# === PILHA DE CHUNKS PARA SUPORTE A RECURSÃO ===
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
def match_all_phrases(tokens, phrase_memory):
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
            test_bigrams = tuple((token_pos_sequence[j], token_pos_sequence[j + 1]) for j in range(i, i + phrase_len - 1))
            if test_bigrams == phrase:
                phrase_str = " ".join([token_pos_sequence[i]] + [token_pos_sequence[i + k + 1] for k in range(len(phrase))])
                matched_phrases.append((phrase, phrase_str))
    return matched_phrases

# === RECONHECIMENTO ===
def recognize_patterns(token_matrix, token_transitions, bigram_memory, phrase_memory, phrase_hierarchy, pos_dict, noise_probability=0.0):
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

                print(f"  → Testando: {prev_token} → {curr_token} | válido? {is_valid_transition}, esperado? {top_down_match}")

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
                matched_phrases = match_all_phrases(flat_pos, phrase_memory)
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
        matched_phrases = match_all_phrases(flat, phrase_memory)
        if matched_phrases:
            chunk_stack.push_chunk(" ".join(flat_pos))
        for _, phrase_str in matched_phrases:
            chunk_stack.push_chunk(phrase_str)
        patterns.append((flat, matched_phrases))

    print("\n🔍 Ligações de Nível 3 (sujeito + verbo):")
    for link in level3_links:
        print(f"   🔗 {link}")

    return patterns

# === EXECUÇÃO ===
if __name__ == "__main__":
    random.seed(42)
    multiwords = load_multiwords(MULTIWORDS_FILE)
    sentence = replace_multiwords(TEST_SENTENCE.lower(), multiwords)
    token_matrix = tokenize(sentence)

    print("\n📌 Tokens da sentença:")
    print(token_matrix)

    token_transitions, bigram_memory, phrase_memory, phrase_hierarchy = load_model(MODEL_FILE)
    pos_dict = load_pos_dictionary(POS_DICTIONARY_FILE)

    print("\n🧠 Executando reconhecimento HTPC com L4 paralelo e multi-token")
    patterns = recognize_patterns(token_matrix, token_transitions, bigram_memory, phrase_memory, phrase_hierarchy, pos_dict, NOISE_PROBABILITY)

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
