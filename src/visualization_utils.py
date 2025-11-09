import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# -------------------------------
# 3D Mesh Visualization
# -------------------------------
def plot_mesh(vertices, title="3D Mesh", save_path=None):
    """Plots mesh vertices as a 3D scatter plot."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -------------------------------
# Error Metric Plots
# -------------------------------
def plot_error_bars(mm_mse, us_mse, mm_mae, us_mae, mesh_name, out_dir):
    """Creates side-by-side bar charts comparing MSE and MAE for each normalization."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = ["X", "Y", "Z"]

    # MSE Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(axes, mm_mse, color='skyblue', label="Min–Max")
    ax[0].set_title("Min–Max MSE per Axis")
    ax[0].set_ylabel("MSE")

    ax[1].bar(axes, us_mse, color='orange', label="Unit Sphere")
    ax[1].set_title("Unit Sphere MSE per Axis")
    plt.suptitle(f"{mesh_name} - Mean Squared Error")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mesh_name}_mse.png")
    plt.close(fig)

    # MAE Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(axes, mm_mae, color='lightgreen', label="Min–Max")
    ax[0].set_title("Min–Max MAE per Axis")
    ax[0].set_ylabel("MAE")

    ax[1].bar(axes, us_mae, color='coral', label="Unit Sphere")
    ax[1].set_title("Unit Sphere MAE per Axis")
    plt.suptitle(f"{mesh_name} - Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mesh_name}_mae.png")
    plt.close(fig)


# -------------------------------
# Comparison Visualization
# -------------------------------
def compare_meshes(original, reconstructed, title, save_path=None):
    """Displays original vs reconstructed meshes for visual comparison."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1, c='blue', label='Original')
    ax1.set_title("Original Mesh")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], s=1, c='red', label='Reconstructed')
    ax2.set_title("Reconstructed Mesh")
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
