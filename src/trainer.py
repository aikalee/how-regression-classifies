from dataset import make_dataset
from losses import mse_loss
from model import SimpleNN
from optimizer import SGDNesterov
from pathlib import Path
import argparse
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Trainer:
    def __init__(self, data, epochs, lr, momentum):
        self.model = SimpleNN()
        self.optimizer = SGDNesterov([self.model.W1, self.model.b1, self.model.W2, self.model.b2], lr=lr, momentum=momentum)
        self.X, self.y = data
        self.epochs = epochs
        self.logged_logits = None
    
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
        
        self.logged_logits = logits
    
    def save_logits(self, save_dir):
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {save_dir}")
        np.save(save_dir / f"logged_logits.npy", np.array(self.logged_logits))
        print("Logits saved at:", save_dir)
    
    def save_checkpoint(self, save_dir):
        params = self.model.params() 

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {save_dir}")

        np.savez(save_dir / f"2d_epoch_{self.epochs}.npz", **params)
        print("Checkpoints saved at:", save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--save_checkpoint_dir", type=Path, default=PROJECT_ROOT / "outputs" / "checkpoints")
    parser.add_argument("--no_save_logits", action="store_true")
    parser.add_argument("--save_logit_dir", type=Path, default=PROJECT_ROOT / "outputs" / "logs")
    args = parser.parse_args()
    data = make_dataset(seed=args.data_seed)
    trainer = Trainer(data, epochs=args.epochs, lr=args.lr, momentum=args.momentum)
    trainer.train()
    if not args.no_save_logits:
        trainer.save_logits(save_dir=args.save_logit_dir)
    if args.save_checkpoints:
        trainer.save_checkpoint(save_dir=args.save_checkpoint_dir)

if __name__ == "__main__":
    main()


