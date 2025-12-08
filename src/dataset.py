from sklearn.datasets import make_blobs
import numpy

def make_dataset(n=800):
    X, y = make_blobs(
        n_samples=n,
        centers=[[-4, 0], [0, 0], [4, 0]],
        cluster_std=0.8,
        random_state=42
    )
    return X, y
