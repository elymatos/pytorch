# Instalar as bibliotecas necessárias (execute no terminal, se não tiver instalado)
# pip install pandas sentence-transformers torch

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# 🔗 === Carregar os dados ===
# Arquivos necessários:
# - frames_sem_cenario.csv  → Deve conter: idFrame, name, description
# - scenario_frames_FN5_complete.csv  → Deve conter: Scenario Frame, Description

# 🔄 Ajuste os caminhos para seus arquivos
frames_sem_cenario = pd.read_csv('frames_sem_cenario.csv')
cenarios = pd.read_csv('scenario_frames_FN5_complete.csv')

# 🔢 Preparar os textos para embeddings
frame_texts = [
    f"{row['name']}: {row['description']}"
    for _, row in frames_sem_cenario.iterrows()
]

cenario_texts = [
    f"{row['Scenario Frame']}: {row['Description']}"
    for _, row in cenarios.iterrows()
]

# 🚀 === Gerar embeddings ===
print("🔄 Gerando embeddings...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

emb_frames = model.encode(frame_texts, convert_to_tensor=True)
emb_cenarios = model.encode(cenario_texts, convert_to_tensor=True)

# 🧠 === Calcular Similaridades ===
print("🔍 Calculando similaridades...")
cosine_scores = util.cos_sim(emb_frames, emb_cenarios)

# 📄 === Gerar Relatório ===
relatorio = []

for idx, frame in enumerate(frames_sem_cenario.itertuples()):
    # Pegar os top 3 matches
    top_matches = torch.topk(cosine_scores[idx], k=3)

    for rank, (score, idx_cenario) in enumerate(zip(top_matches.values, top_matches.indices)):
        cenario = cenarios.iloc[idx_cenario.item()]
        relatorio.append({
            "idFrame": frame.idFrame,
            "Frame": frame.name,
            "Frame Description": frame.description,
            "Rank": rank + 1,
            "Cenário Sugerido": cenario["Scenario Frame"],
            "Cenário Descrição": cenario["Description"],
            "Score de Similaridade": round(score.item(), 4),
            "Justificativa": f"Frame '{frame.name}' apresenta maior similaridade semântica com o cenário '{cenario['Scenario Frame']}', cuja descrição é: {cenario['Description']}"
        })

# 🔽 === Salvar o Relatório ===
relatorio_df = pd.DataFrame(relatorio)
relatorio_df.to_csv('relatorio_alocacao_frames_sem_cenario.csv', index=False, encoding='utf-8-sig')

print("✅ Relatório gerado: relatorio_alocacao_frames_sem_cenario.csv")
