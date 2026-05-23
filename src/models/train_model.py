import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks


def build_model(input_dim):
    """Builds a simple feed‑forward binary classifier."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Trains the model with early stopping and returns the trained model + history."""
    model = build_model(X_train.shape[1])

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es]
    )

    return model, history
