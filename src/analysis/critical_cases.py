import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def select_critical_cases(preds_df):
    """Selects strong positives, misclassifications, and borderline cases per class."""

    # Strong positives
    strong_positive_0 = preds_df[preds_df.true == 0].sort_values("prob").iloc[0]
    strong_positive_1 = preds_df[preds_df.true == 1].sort_values("prob", ascending=False).iloc[0]

    # Misclassifications
    misclass_0 = preds_df[(preds_df.true == 0) & (preds_df.pred == 1)].iloc[0]
    misclass_1 = preds_df[(preds_df.true == 1) & (preds_df.pred == 0)].iloc[0]

    # Borderline (closest to 0.5)
    borderline_0 = preds_df[preds_df.true == 0].iloc[
        (preds_df[preds_df.true == 0].prob - 0.5).abs().argmin()
    ]
    borderline_1 = preds_df[preds_df.true == 1].iloc[
        (preds_df[preds_df.true == 1].prob - 0.5).abs().argmin()
    ]

    selected = [
        strong_positive_0.patient_id,
        strong_positive_1.patient_id,
        misclass_0.patient_id,
        misclass_1.patient_id,
        borderline_0.patient_id,
        borderline_1.patient_id
    ]

    return selected


def nearest_neighbour_pairs(X, preds_df, n_neighbors=5):
    """Finds NN pairs with different predictions or different true labels."""
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    pairs = []

    for i, neighs in enumerate(indices):
        for k, j in enumerate(neighs[1:]):  # skip self
            dist = distances[i][k + 1]

            if preds_df.iloc[i].pred != preds_df.iloc[j].pred:
                pairs.append(("diff_predicted", preds_df.iloc[i].patient_id,
                              preds_df.iloc[j].patient_id, dist))

            elif preds_df.iloc[i].true != preds_df.iloc[j].true:
                if preds_df.iloc[i].pred == preds_df.iloc[i].true and preds_df.iloc[j].pred == preds_df.iloc[j].true:
                    pairs.append(("diff_true_correct", preds_df.iloc[i].patient_id,
                                  preds_df.iloc[j].patient_id, dist))
                else:
                    pairs.append(("diff_true_incorrect", preds_df.iloc[i].patient_id,
                                  preds_df.iloc[j].patient_id, dist))

    return pairs
