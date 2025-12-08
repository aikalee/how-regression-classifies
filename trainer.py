from dataset import make_dataset
from losses import mse_loss
from model import SimpleNN
from optimizer import SGDNesterov

import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Trainer:
    def __init__(self, data, lr=0.05, momentum=0.9):
        self.model = SimpleNN()
        self.optimizer = SGDNesterov(self.model.params(), lr=lr, momentum=momentum)
        self.X, self.y = data
        self.logged_logits = []
    
    def train(self, epochs=100, save_logits=False):
        for epoch in range(1, epochs+1):
            logits = self.model.forward(self.X)
            if epoch % 10 == 0:
                self.logged_logits.append(logits.copy())
            loss, grad_out = mse_loss(logits, self.y)
            grads = self.model.backward(grad_out)
            self.optimizer.step(grads)

            if epoch % 10 == 0:
                pred = self.model.predict(self.X)
                acc = (pred == self.y).mean()
                print(f"epoch {epoch}, loss={loss:.4f}, acc={acc:.3f}")

        if save_logits:
            np.save(os.path.join(PROJECT_ROOT, "logits_history.npy"), np.array(self.logged_logits))
            print("Logits saved at:", os.path.join(PROJECT_ROOT, "logits_history.npy"))


def main():
    data = make_dataset()
    trainer = Trainer(data)
    trainer.train(epochs=100, save_logits=True)

if __name__ == "__main__":
    main()


