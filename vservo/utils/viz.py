import matplotlib.pyplot as plt
import numpy as np

def plot_error_history(error_history: np.ndarray):
    """Plot error vs. iteration for a given error history array."""
    iterations = np.arange(1, len(error_history) + 1)
    plt.figure()
    plt.plot(iterations, error_history, marker='o')
    plt.title('Error vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def plot_velocity_history(data):
    """
    Plot each column of a 2D array vs iteration index.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Matrix to plot where each column is plotted as a line.
    """
    data = np.asarray(data)
    n_cols = data.shape[1]
    plt.figure(figsize=(10, 6))
    for i in range(n_cols):
        plt.plot(data[:, i], label=f'Column {i + 1}')
    plt.title("Matrix Column Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
