import numpy as np

class SimpleNN:
    def __init__(self, in_dim=1, hidden=16, out_dim=3):
        self.W1 = np.random.randn(in_dim, hidden) * np.sqrt(1 / in_dim)
        self.b1 = np.zeros((1, hidden))

        self.W2 = np.random.randn(hidden, out_dim) * np.sqrt(1 / hidden)
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
        self.z1 = X @ self.W1 + self.b1
        self.h1 = np.tanh(self.z1)

        self.z2 = self.h1 @ self.W2 + self.b2
        self.out = self.z2
        
        return self.out

    def backward(self, grad_out): 
        # layer 2 (output)
        dW2 = self.h1.T @ grad_out
        db2 = grad_out.sum(axis=0, keepdims=True)

        dh1 = grad_out @ self.W2.T
        dh1_raw = dh1 * (1 - self.h1**2)     # tanh'

        # layer 1
        dW1 = self.X.T @ dh1_raw
        db1 = dh1_raw.sum(axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(axis=1)
