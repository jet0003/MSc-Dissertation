import pandas as pd
from src.models.ensemble import load_fold_models, EnsembleWrapper
from src.explainability.shap_utils import compute_shap_values
from src.explainability.permutation_importance import compute_permutation_importance
from src.explainability.integrated_gradients import integrated_gradients


def run_explainability(X_train_path, X_test_path, ids_test_path, metrics_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    ids_test = pd.read_csv(ids_test_path)["patient_id"]
    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()

    models = load_fold_models()
    ensemble = EnsembleWrapper(models)

    # SHAP
    shap_df, shap_values, explainer = compute_shap_values(
        ensemble, X_train, X_test, ids_test
    )
    shap_df.to_csv("ensemble_shap_values.csv", index=False)

    # Permutation importance
    perm_df = compute_permutation_importance(
        ensemble, X_test, shap_df["true"], metrics, ensemble=True
    )
    perm_df.to_csv("permutation_importance_ensemble.csv", index=False)

    # Integrated Gradients
    ig = integrated_gradients(ensemble.models[0], X_test.values)
    pd.DataFrame(ig, columns=X_test.columns).to_csv("integrated_gradients.csv", index=False)
