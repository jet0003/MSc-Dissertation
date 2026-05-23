import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_permutation_importance(model, X, y, metric_baseline, ensemble=False):
    """Permutation importance for single model or ensemble."""
    results = []

    for col in X.columns:
        X_perm = X.copy()
        X_perm[col] = np.random.permutation(X_perm[col])

        if ensemble:
            y_pred = model.predict(X_perm)[:, 1]
        else:
            y_pred = model.predict(X_perm).flatten()

        auc = roc_auc_score(y, y_pred)
        drop = metric_baseline["auc"] - auc

        results.append({"feature": col, "auc_drop": drop})

    return pd.DataFrame(results)
