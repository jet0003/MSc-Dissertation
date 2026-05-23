import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)


def evaluate_model(y_true, y_pred_proba, ids, fold_num="fold"):
    """Evaluates a model and returns predictions dataframe + metrics dict."""
    y_pred = (y_pred_proba >= 0.5).astype(int)

    preds_df = pd.DataFrame({
        "patient_id": ids,
        "true": y_true,
        "pred": y_pred,
        "prob": y_pred_proba
    })

    metrics = {
        "fold": fold_num,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

    return preds_df, metrics
