import pandas as pd
import networkx as nx


def build_correlation_graph(X, threshold=0.3, drop_cols=None):
    """Builds a graph where edges represent |Spearman correlation| > threshold."""
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    corr = X.corr(method="spearman")
    G = nx.Graph()

    for i, f1 in enumerate(corr.columns):
        for j, f2 in enumerate(corr.columns):
            if i < j:
                val = corr.iloc[i, j]
                if abs(val) > threshold:
                    G.add_edge(f1, f2, weight=val)

    return G


def second_degree_pairs(G):
    """Returns feature pairs within 2 hops."""
    pairs = []
    for f1 in G.nodes():
        for f2 in G.nodes():
            if f1 != f2:
                try:
                    d = nx.shortest_path_length(G, f1, f2)
                    if d <= 2:
                        pairs.append((f1, f2, d))
                except nx.NetworkXNoPath:
                    pass
    return pairs
