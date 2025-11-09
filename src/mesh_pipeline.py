from normalization_utils import (
    minmax_normalize,
    unit_sphere_normalize,
    quantize,
    dequantize,
    compute_errors
)

from visualization_utils import (
    plot_error_bars,
    plot_mesh,
    compare_meshes

)import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Utility Functions
# -----------------------------
def load_obj_vertices(path):
    """Loads vertex coordinates (v x y z) from a .obj file."""
    vertices = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = map(float, parts[1:4])
                        vertices.append([x, y, z])
                    except ValueError:
                        continue
    return np.array(vertices, dtype=float)


def save_ply_vertices(path, vertices):
    """Saves vertices as a .ply file (for easy visualization)."""
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")


def minmax_normalize(vertices):
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    denom = np.where(vmax - vmin == 0, 1, vmax - vmin)
    normalized = (vertices - vmin) / denom
    return normalized, vmin, vmax


def unit_sphere_normalize(vertices):
    centroid = vertices.mean(axis=0)
    shifted = vertices - centroid
    max_dist = np.linalg.norm(shifted, axis=1).max()
    if max_dist == 0:
        max_dist = 1.0
    normalized = shifted / max_dist
    return normalized, centroid, max_dist


def quantize(normalized, bins=1024):
    q = np.floor(normalized * (bins - 1) + 0.5).astype(int)
    return np.clip(q, 0, bins - 1)


def dequantize(q, bins=1024):
    return q.astype(float) / (bins - 1)


def compute_errors(original, reconstructed):
    diff = original - reconstructed
    mse = np.mean(diff ** 2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)
    return mse, mae


# -----------------------------
# Main Processing Function
# -----------------------------
def process_meshes(input_dir, output_dir, bins=1024):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for mesh_file in sorted(input_dir.glob("*.obj")):
        print(f"\n🔹 Processing {mesh_file.name}...")
        vertices = load_obj_vertices(mesh_file)

        if len(vertices) == 0:
            print("⚠️ No vertices found — skipping.")
            continue

        print(f"   Total vertices: {len(vertices)}")

        # Task 1: Mesh Statistics
        v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
        v_mean, v_std = vertices.mean(axis=0), vertices.std(axis=0)
        print(f"   Min: {v_min}")
        print(f"   Max: {v_max}")
        print(f"   Mean: {v_mean}")
        print(f"   Std: {v_std}")

        # -------------------------
        # Task 2: Normalization + Quantization
        # -------------------------
        # Min–Max Normalization
        mm_norm, mm_min, mm_max = minmax_normalize(vertices)
        mm_q = quantize(mm_norm, bins)
        mm_deq = dequantize(mm_q, bins)
        mm_recon = mm_deq * (mm_max - mm_min) + mm_min
        mm_ply = output_dir / f"{mesh_file.stem}_minmax.ply"
        save_ply_vertices(mm_ply, mm_recon)

        # Unit Sphere Normalization
        us_norm, us_center, us_scale = unit_sphere_normalize(vertices)
        us_norm01 = (us_norm + 1.0) / 2.0
        us_q = quantize(us_norm01, bins)
        us_deq01 = dequantize(us_q, bins)
        us_deq = us_deq01 * 2.0 - 1.0
        us_recon = us_deq * us_scale + us_center
        us_ply = output_dir / f"{mesh_file.stem}_unitsphere.ply"
        save_ply_vertices(us_ply, us_recon)

        # -------------------------
        # Task 3: Error Analysis
        # -------------------------
        mm_mse, mm_mae = compute_errors(vertices, mm_recon)
        us_mse, us_mae = compute_errors(vertices, us_recon)

        axes = ['X', 'Y', 'Z']

        # MSE Plot
        plt.figure(figsize=(8, 4))
        plt.bar(axes, mm_mse, label='Min–Max')
        plt.bar(axes, us_mse, bottom=mm_mse, alpha=0.6, label='Unit Sphere')
        plt.title(f"MSE Comparison – {mesh_file.stem}")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.tight_layout()
        mse_plot = output_dir / f"{mesh_file.stem}_mse.png"
        plt.savefig(mse_plot)
        plt.close()

        # MAE Plot
        plt.figure(figsize=(8, 4))
        plt.bar(axes, mm_mae, label='Min–Max')
        plt.bar(axes, us_mae, bottom=mm_mae, alpha=0.6, label='Unit Sphere')
        plt.title(f"MAE Comparison – {mesh_file.stem}")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.tight_layout()
        mae_plot = output_dir / f"{mesh_file.stem}_mae.png"
        plt.savefig(mae_plot)
        plt.close()

        summary.append({
            "file": mesh_file.name,
            "vertices": len(vertices),
            "minmax_mse": mm_mse.tolist(),
            "minmax_mae": mm_mae.tolist(),
            "unitsphere_mse": us_mse.tolist(),
            "unitsphere_mae": us_mae.tolist(),
            "mse_plot": mse_plot.as_posix(),
            "mae_plot": mae_plot.as_posix(),
        })

    # Save Summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Processing Complete!")
    print(f"📁 Outputs saved to: {output_dir}")
    print(f"📄 Summary file: {summary_path}")


# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/mesh_pipeline.py <input_dir> [output_dir]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(input_dir, "outputs")
    process_meshes(input_dir, output_dir)
  
