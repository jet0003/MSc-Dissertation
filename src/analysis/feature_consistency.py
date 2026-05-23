import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kruskal


def compute_rank_stability(rankings_df):
    """Computes fold-by-fold Spearman rank correlations."""
    folds = rankings_df["fold"].unique()
    features = rankings_df["feature"].unique()

    pivot = rankings_df.pivot(index="feature", columns="fold", values="rank")

    corr_matrix = pivot.corr(method="spearman")
    return corr_matrix


def consensus_ranking(rankings_df):
    """Computes mean rank + std rank across folds."""
    grouped = rankings_df.groupby("feature")["rank"].agg(["mean", "std"])
    grouped = grouped.sort_values("mean")
    grouped.columns = ["rank_mean", "rank_std"]
    return grouped


def kruskal_shap(shap_df):
    """Kruskal–Wallis test for SHAP stability across folds."""
    results = []

    for feature in shap_df.columns:
        if feature in ["patient_id", "fold"]:
            continue

        groups = [
            shap_df[shap_df.fold == f][feature].values
            for f in shap_df.fold.unique()
        ]

        stat, p = kruskal(*groups)
        results.append({"feature": feature, "p_value": p})

    return pd.DataFrame(results)
