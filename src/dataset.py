from sklearn.datasets import make_blobs
import numpy as np

import numpy as np

def make_dataset(N=600, means=(-2.0, 0.0, 2.0), std=0.3, seed=None):
    """
    Create a simple 1D 3-class dataset using Gaussian clusters.
    
    Returns:
        X : (N, 1) array of input values
        y : (N,) array of integer class labels {0,1,2}
    """

    if seed is not None:
        np.random.seed(seed)

    n_per_class = N // 3

    x0 = np.random.normal(means[0], std, n_per_class)
    x1 = np.random.normal(means[1], std, n_per_class)
    x2 = np.random.normal(means[2], std, n_per_class)

    # Combine into one dataset
    X = np.concatenate([x0, x1, x2])[:, None]   # shape (N, 1)
    y = np.concatenate([
        np.zeros(n_per_class, int),
        np.ones(n_per_class, int),
        np.full(n_per_class, 2, int),
    ])

    # Shuffle
    idx = np.random.permutation(N)
    return X[idx], y[idx]


# def make_dataset(n=800, x_range=(-6, 6), noise=0.0, boundary=0.3,seed=None):
#     """
#     Realistic 1D -> 3-class dataset.
#     Classes are determined by a hidden nonlinear function:
#         f(x) = sin(2x)
#     and thresholded into 3 regions.
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     # Sample X
#     X = np.random.uniform(x_range[0], x_range[1], n)

#     # Hidden function
#     f = np.sin(2 * X)

#     # Base labels before noise
#     Y = np.zeros(n, dtype=int)
#     Y[(f >= -boundary) & (f <= boundary)] = 1
#     Y[f > boundary] = 2

#     # Add noise by randomly flipping labels
#     if noise > 0:
#         flip_mask = np.random.rand(n) < noise
#         Y[flip_mask] = np.random.randint(0, 3, flip_mask.sum())

#     # Shuffle
#     idx = np.random.permutation(n)
#     return X[idx][:, None], Y[idx]

def main():
    print(make_dataset(n=800, noise=0.05, boundary=0.3, seed=42))

if __name__ == "__main__":
    main()
