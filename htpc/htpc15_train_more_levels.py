# HTPC Training Script with Phrase Hierarchy
import json
from itertools import product
from collections import defaultdict
from datetime import datetime

INPUT_FILE = "training_sentences_long.txt"
BLACKLIST_FILE = "blacklisted_bigrams.txt"
MULTIWORDS_FILE = "multiwords.txt"
OUTPUT_FILE = "htpc_model.json"
PHRASE_CHUNK_SIZE = 3

def normalize_token(token):
    return token.lower().strip(".,!?;:()[]{}\"'")

def tokenize(sentence):
    tokens = []
    for word in sentence.strip().split():
        options = [normalize_token(w) for w in word.split('|')]
        tokens.append(options)
    return tokens

def expand_sequences(token_matrix):
    return list(product(*token_matrix))

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

def load_blacklist(path):
    blacklist = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = normalize_token(line.strip()).split()
            if len(parts) == 2:
                blacklist.add(tuple(parts))
    return blacklist

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
                level2["|||".join([f"{a}__{b}" for a, b in phrase_a])].append(" ".join(combined))
    return level2

def convert_for_json(model, hierarchy):
    return {
        'metadata': {
            'trained_on': datetime.now().isoformat(),
            'num_sentences': model['num_sentences'],
            'vocab_size': len(model['vocab']),
        },
        'token_transitions': model['token_transitions'],
        'bigram_memory': {
            f"{k[0]}|||{k[1]}": v for k, v in model['bigram_memory'].items()
        },
        'phrase_memory': {
            "|||".join([f"{a}__{b}" for (a, b) in k]): v
            for k, v in model['phrase_memory'].items()
        },
        'phrase_hierarchy': hierarchy
    }

def train_htpc_extended(input_path, blacklist_path, multiwords_path, output_path, chunk_size=3):
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
        'token_transitions': token_transitions,
        'bigram_memory': bigram_memory,
        'phrase_memory': phrase_memory,
        'vocab': vocab,
        'num_sentences': len(expanded_sequences)
    }

    json_model = convert_for_json(model, phrase_hierarchy)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_model, f, indent=2, ensure_ascii=False)

    print(f"✅ Modelo treinado com {len(expanded_sequences)} sequências.")
    print(f"📘 Vocabulário: {len(vocab)} tokens.")
    print(f"🧠 Frases compostas armazenadas: {len(phrase_hierarchy)}")
    print(f"💾 Salvo em: {output_path}")

if __name__ == "__main__":
    # Executa o treinamento e imprime a hierarquia de frases
    train_htpc_extended(INPUT_FILE, BLACKLIST_FILE, MULTIWORDS_FILE, OUTPUT_FILE, chunk_size=PHRASE_CHUNK_SIZE)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        model = json.load(f)

    phrase_hierarchy = model.get("phrase_hierarchy", {})

    print("🔍 Exemplos de frases compostas (nível 2):")
    for base_key, continuations in phrase_hierarchy.items():
        print(f"  {base_key} →")
        for cont in continuations:
            print(f"     → {cont}")

    print("🔍 Exemplos de frases compostas (nível 2):")
    for base_key, continuations in phrase_hierarchy.items():
        print(f"  {base_key} →")
        for cont in continuations:
            print(f"     → {cont}")
