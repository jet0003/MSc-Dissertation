import numpy as np
import tensorflow as tf


class EnsembleWrapper:
    """Wraps multiple fold models and averages their predictions."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [m.predict(X, verbose=0) for m in self.models]
        avg = np.mean(preds, axis=0)
        return np.hstack([1 - avg, avg])  # shape (n, 2)


def load_fold_models(k_folds=5):
    """Loads saved fold models model_1.keras ... model_k.keras."""
    models = []
    for i in range(1, k_folds + 1):
        m = tf.keras.models.load_model(f"model_{i}.keras")
        models.append(m)
    return models
