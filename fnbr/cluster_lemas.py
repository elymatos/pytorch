import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import json
from tqdm import tqdm

# =======================
# Carregar os lemas
# =======================
file_path = "lu_candidate_dedupe.csv"
df = pd.read_csv(file_path)
df.columns = ['lema']
lemmas = df['lema'].dropna().unique().tolist()

# =======================
# Identificar verbos
verbos = [l for l in lemmas if re.match(r'.*(ar|er|ir)$', l)]

print(f"Número de verbos identificados: {len(verbos)}")

# =======================
# Carregar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# =======================
# Gerar embeddings
print("Calculando embeddings...")
lemma_embeddings = model.encode(lemmas, convert_to_tensor=True, show_progress_bar=True)
verb_embeddings = model.encode(verbos, convert_to_tensor=True, show_progress_bar=True)

# =======================
# Encontrar clusters
clusters = {}

for verbo, verbo_emb in tqdm(zip(verbos, verb_embeddings), total=len(verbos), desc="Processando verbos"):
    similarities = util.cos_sim(verbo_emb, lemma_embeddings)[0]
    related_indices = (similarities >= 0.85).nonzero().squeeze().tolist()
    if isinstance(related_indices, int):
        related_indices = [related_indices]
    related_words = [lemmas[i] for i in related_indices]
    clusters[verbo] = related_words

# =======================
# Salvar resultado
with open("lexical_clusters_todos_verbos.json", "w", encoding="utf-8") as f:
    json.dump(clusters, f, ensure_ascii=False, indent=2)

print("Clusters salvos em lexical_clusters_todos_verbos.json")
