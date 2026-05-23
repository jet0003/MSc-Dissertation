import numpy as np
import tensorflow as tf


def integrated_gradients(model, x, baseline=None, steps=64):
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps)
    alphas_x = baseline[None, ...] + alphas[:, None, None] * (x[None, ...] - baseline[None, ...])

    with tf.GradientTape() as tape:
        tape.watch(alphas_x)
        preds = model(tf.reshape(alphas_x, (-1, x.shape[1])))
        preds_tf = tf.reshape(preds, (steps, x.shape[0], -1))[:, :, 0]
        preds_mean = tf.reduce_mean(preds_tf, axis=0)

    grads = tape.gradient(preds_mean, alphas_x)
    avg_grads = tf.reduce_mean(grads, axis=0)

    ig = (x - baseline) * avg_grads
    return ig.numpy()
