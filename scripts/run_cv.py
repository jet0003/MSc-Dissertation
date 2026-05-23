import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

from src.models.train_model import train_model
from src.models.evaluate import evaluate_model


def run_cv(X, y, ids, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    all_preds = []
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        ids_val = ids.iloc[val_idx]

        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)

        model, _ = train_model(X_res, y_res, X_val, y_val)
        model.save(f"model_{fold}.keras")

        y_pred_proba = model.predict(X_val).flatten()
        preds_df, metrics = evaluate_model(y_val, y_pred_proba, ids_val, fold_num=fold)

        all_preds.append(preds_df)
        all_metrics.append(metrics)

    pd.concat(all_preds).to_csv("all_folds_predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv("cv_metrics_summary.csv", index=False)
