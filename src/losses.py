import numpy as np

def mse_loss(logits, y):
    y_onehot = np.eye(3)[y]
    loss = ((logits - y_onehot)**2).mean()
    grad_out = 2 * (logits - y_onehot) / y_onehot.size
    return loss, grad_out
