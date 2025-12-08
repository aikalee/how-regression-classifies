from dataset import make_dataset
from losses import mse_loss
from model import SimpleNN
from optimizer import SGDNesterov
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Trainer:
    def __init__(self, data, epochs=100, lr=0.05, momentum=0.9):
        self.model = SimpleNN()
        self.optimizer = SGDNesterov([self.model.W1, self.model.b1, self.model.W2, self.model.b2], lr=lr, momentum=momentum)
        self.X, self.y = data
        self.epochs = epochs
    
    def train(self):
        for epoch in range(1, self.epochs+1):
            logits = self.model.forward(self.X)
            
            loss, grad_out = mse_loss(logits, self.y)
            grads = self.model.backward(grad_out)
            self.optimizer.step(grads)

            if epoch % 10 == 0:
                pred = self.model.predict(self.X)
                acc = (pred == self.y).mean()
                print(f"epoch {epoch}, loss={loss:.4f}, acc={acc:.3f}")
    
    def save_checkpoint(self, save_dir=PROJECT_ROOT / "outputs" / "checkpoints"):
        params = self.model.params() 

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {save_dir}")

        np.savez(save_dir / f"2d_epoch_{self.epochs}.npz", **params)
        print("Checkpoints saved at:", save_dir)


def main():
    data = make_dataset()
    trainer = Trainer(data)
    trainer.train()
    trainer.save_checkpoint()

if __name__ == "__main__":
    main()


