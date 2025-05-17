#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import sys

def main(input_path, output_path):
    # 1. Defina o mapeamento de tags para agrupamento
    tag_mapping = {
        'PRON': 'N',
        'NOUN': 'N',
        'PROPN': 'N',
        'AUX': 'V',
        'VERB': 'V'
        # as demais tags permanecem inalteradas, ou mapeie conforme desejar
    }

    # 2. Leia e agrupe as sentenças POS
    sentences = []
    with open(input_path, 'r') as f:
        for line in f:
            tags = [tag_mapping.get(tag, tag) for tag in line.strip().split()]
            if tags:
                sentences.append(tags)

    if not sentences:
        print("Nenhuma sentença encontrada em", input_path)
        return

    # 3. Construa o grafo dirigido com as tags agrupadas
    G = nx.DiGraph()
    for sent in sentences:
        for src, dst in zip(sent, sent[1:]):
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)

    # 4. Normalize pesos para probabilidades
    for node in G.nodes():
        total = sum(G[node][nbr]['weight'] for nbr in G[node])
        for nbr in G[node]:
            G[node][nbr]['prob'] = G[node][nbr]['weight'] / total

    # 5. Desenho da rede
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    # Nós
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightgreen')

    # Arestas com largura proporcional à probabilidade
    edge_widths = [d['prob'] * 5 for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=15)

    # Rótulos dos nós
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Rótulos das arestas (probabilidades)
    edge_labels = {(u, v): f"{d['prob']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Rede Semântica de POS Tags (Agrupadas)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Rede salva em {output_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python build_pos_network_grouped.py <input.txt> <output.png>")
    else:
        main(sys.argv[1], sys.argv[2])
