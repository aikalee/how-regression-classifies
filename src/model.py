import numpy as np

class SimpleNN:
    def __init__(self, in_dim=2, hidden=16, out_dim=3):
        self.W1 = np.random.randn(in_dim, hidden) * 0.1
        self.b1 = np.zeros((1, hidden))

        self.W2 = np.random.randn(hidden, out_dim) * 0.1
        self.b2 = np.zeros((1, out_dim))
    
    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params):
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
    
    def forward(self, X):
        self.X = X
        self.h = np.tanh(X @ self.W1 + self.b1)
        self.out = self.h @ self.W2 + self.b2  # regression heads
        return self.out

    def backward(self, grad_out):
        # dLoss/dW2
        dW2 = self.h.T @ grad_out
        db2 = grad_out.sum(axis=0, keepdims=True)

        # Backprop into hidden layer
        dh = grad_out @ self.W2.T
        dh_raw = dh * (1 - self.h**2)

        # dLoss/dW1
        dW1 = self.X.T @ dh_raw
        db1 = dh_raw.sum(axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(axis=1)
