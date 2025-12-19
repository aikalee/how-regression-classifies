import numpy as np

def mse_loss(logits, y):
    N = logits.shape[0]
    y_onehot = np.eye(logits.shape[1])[y]   # (N, C)

    diff = logits - y_onehot
    loss = 0.5 * np.mean(np.sum(diff**2, axis=1))

    grad_out = diff / N

    return loss, grad_out