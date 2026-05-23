import shap
import numpy as np
import pandas as pd


def compute_shap_values(model, X_background, X_eval, ids):
    """Computes SHAP values for a model on evaluation data."""
    explainer = shap.KernelExplainer(model.predict, X_background)
    shap_values = explainer.shap_values(X_eval)

    df = pd.DataFrame(shap_values, columns=X_eval.columns)
    df["patient_id"] = ids

    return df, shap_values, explainer


def plot_shap_bar(shap_values, feature_names, title="SHAP Summary"):
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar")


def plot_shap_beeswarm(shap_values, X_eval):
    shap.summary_plot(shap_values, X_eval)
