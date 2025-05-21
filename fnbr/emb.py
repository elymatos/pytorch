# ==========================
# Script para Clusterização Semântica de Lemas
# ==========================

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors
from tqdm import tqdm

# ========= CONFIGURAÇÕES =========

# Arquivo de entrada com lemas
INPUT_FILE = 'lu_candidate.csv'

# Arquivo de saída com clusters
OUTPUT_FILE = 'clusters_output.csv'

# Caminho do modelo fastText pré-treinado em português (.bin)
# Exemplo de modelo: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz
FASTTEXT_MODEL_PATH = 'cc.pt.300.bin'

# Número de clusters (pode ser ajustado)
N_CLUSTERS = 4000

# ========= LEITURA DOS DADOS =========

print("Lendo o arquivo de lemas...")
lemmas_df = pd.read_csv(INPUT_FILE)
lemmas_list = lemmas_df.iloc[:, 0].dropna().astype(str).tolist()

# ========= CARREGANDO MODELO =========

print("Carregando modelo fastText...")
model = load_facebook_model(FASTTEXT_MODEL_PATH)
model = model.wv

# ========= GERANDO EMBEDDINGS =========

print("Gerando embeddings...")
embeddings = []
valid_lemmas = []

for lemma in tqdm(lemmas_list):
    try:
        vector = model[lemma.lower()]
        embeddings.append(vector)
        valid_lemmas.append(lemma)
    except KeyError:
        print(f"Lemma não encontrado no modelo: {lemma}")

embeddings = np.array(embeddings)

# ========= CLUSTERIZAÇÃO =========

print(f"Executando clustering com {N_CLUSTERS} clusters...")
clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit(embeddings)

# ========= ORGANIZANDO RESULTADOS =========

print("Organizando os resultados...")
results = pd.DataFrame({
    'Lemma': valid_lemmas,
    'Cluster': clustering.labels_
})

# ========= SALVANDO SAÍDA =========

results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"Arquivo de saída salvo em: {OUTPUT_FILE}")

print("Processo concluído com sucesso!")
