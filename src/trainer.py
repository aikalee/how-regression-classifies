from dataset import make_dataset
from losses import mse_loss
from model import SimpleNN
from optimizer import SGDNesterov
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Trainer:
    def __init__(self, data, epochs=100, lr=0.01, momentum=0.9):
        self.model = SimpleNN()
        self.optimizer = SGDNesterov([self.model.W1, self.model.b1, self.model.W2, self.model.b2, self.model.W3, self.model.b3], lr=lr, momentum=momentum)
        self.X, self.y = data
        self.epochs = epochs
        self.logged_logits = None
    
    def train(self):
        for epoch in range(1, self.epochs+1):
            logits = self.model.forward(self.X)
            loss, grad_out = mse_loss(logits, self.y)
            grads = self.model.backward(grad_out)
            self.optimizer.step(grads)
            # print("mean |W1|:", np.abs(self.model.params()["W1"]).mean())
           
            if epoch % 10 == 0:
                # self.logged_logits.append(logits.copy())
                pred = self.model.predict(self.X)
                acc = (pred == self.y).mean()
                print(f"epoch {epoch}, loss={loss:.4f}, acc={acc:.3f}")
        
        self.logged_logits = logits
    
    def save_logits(self, save_dir=PROJECT_ROOT / "outputs" / "logs"):
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {save_dir}")
        np.save(save_dir / f"logged_logits.npy", np.array(self.logged_logits))
        print("Logits saved at:", save_dir)
    
    def save_checkpoint(self, save_dir=PROJECT_ROOT / "outputs" / "checkpoints"):
        params = self.model.params() 

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {save_dir}")

        np.savez(save_dir / f"2d_epoch_{self.epochs}.npz", **params)
        print("Checkpoints saved at:", save_dir)


def main():
    data = make_dataset(seed=42)
    trainer = Trainer(data, epochs=100)
    trainer.train()
    trainer.save_logits()

if __name__ == "__main__":
    main()


