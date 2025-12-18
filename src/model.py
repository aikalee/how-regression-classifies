import numpy as np

class SimpleNN:
    def __init__(self, in_dim=1, hidden=16, out_dim=3):
        self.W1 = np.random.randn(in_dim, hidden) * np.sqrt(1 / in_dim)
        self.b1 = np.zeros((1, hidden))

        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(1 / hidden)
        self.b2 = np.zeros((1, hidden))

        self.W3 = np.random.randn(hidden, out_dim) * np.sqrt(1 / hidden)
        self.b3 = np.zeros((1, out_dim))
    
    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def set_params(self, params):
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        self.W3 = params["W3"]
        self.b3 = params["b3"]
    
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = np.tanh(self.z1)

        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.tanh(self.z2)

        self.z3 = self.h2 @ self.W3 + self.b3
        self.out = self.z3
        return self.out

    def backward(self, grad_out):
        # layer 3 (output)
        dW3 = self.h2.T @ grad_out
        db3 = grad_out.sum(axis=0, keepdims=True)

        dh2 = grad_out @ self.W3.T
        dh2_raw = dh2 * (1 - self.h2**2)   # tanh'

        # layer 2
        dW2 = self.h1.T @ dh2_raw
        db2 = dh2_raw.sum(axis=0, keepdims=True)

        dh1 = dh2_raw @ self.W2.T
        dh1_raw = dh1 * (1 - self.h1**2)

        # layer 1
        dW1 = self.X.T @ dh1_raw
        db1 = dh1_raw.sum(axis=0, keepdims=True)

        return [dW1, db1, dW2, db2, dW3, db3]

    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(axis=1)
