from pickle import DUP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

from ticl.prediction import MotherNetClassifier
from mixture_hypernetworks import *

# ----------- Data Generation Function -----------

def generate_concentric_rings(
    n_rings=6,
    points_per_ring=200,
    start_radius=1.0,
    ring_width=0.4,
    gap=0.6,
    noise_std=0.05,
    random_state=None,
    imbalance_ratio=0.2,
    majority_class=0
):
    """Generate 2D points arranged in concentric rings of alternating classes."""
    rng = np.random.RandomState(int(random_state)) if random_state else np.random

    if imbalance_ratio <= 0:
        raise ValueError("imbalance_ratio must be positive.")

    base_counts = (
        [int(points_per_ring)] * n_rings
        if np.isscalar(points_per_ring)
        else [int(c) for c in points_per_ring]
    )

    counts = []
    for i in range(n_rings):
        label = i % 2
        base_n = base_counts[i]
        
        if imbalance_ratio == 1.0:
            n = base_n
        else:
            n = int(round(base_n * imbalance_ratio)) if label == majority_class else int(round(base_n))
            
        counts.append(max(1, n))

    X_list, y_list = [], []
    for i in range(n_rings):
        center_radius = start_radius + i * (ring_width + gap)
        n = counts[i]
        angles = rng.uniform(0, 2 * np.pi, size=n)
        radial_offsets = rng.uniform(-ring_width / 2, ring_width / 2, size=n)
        radii = center_radius + radial_offsets
        
        xs = radii * np.cos(angles) + rng.normal(scale=noise_std, size=n)
        ys = radii * np.sin(angles) + rng.normal(scale=noise_std, size=n)
        
        X_list.append(np.column_stack([xs, ys]))
        y_list.append(np.full(n, i % 2, dtype=int))

    return np.vstack(X_list), np.concatenate(y_list)

# ----------- Plotting Function -----------

def plot_decision_regions(
    model, X, y, X_test, y_test,
    is_mixture=False,
    grid_step=0.1,
    padding=0.05,
    ax=None,
    title=None,
    fit=True,
    index=0,
    class0_color="blue",
    class1_color="orangered",
):
    """Train model, create a 2D grid, and plot predicted class regions."""
    X, y = np.asarray(X), np.asarray(y)
    
    if fit:
        if is_mixture:
            model.fit_mixture_hypernetworks(X, y)
        else:
            model.fit(X, y)

    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid and test set
    if not is_mixture:
        Z = model.predict(grid)
        y_pred = model.predict(X_test)
    else:
        _, _, Z, _, _, _, _ = model.predict_2(grid)
        _, _, y_pred, _, _, _, _ = model.predict_2(X_test)

    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    Z = Z.reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    classes = np.unique(y)
    y_plot = np.searchsorted(classes, y)
    Z_plot = np.searchsorted(classes, Z)

    cmap = ListedColormap([class0_color, class1_color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)

    ax.pcolormesh(xx, yy, Z_plot, shading="auto", alpha=0.3, cmap=cmap, norm=norm)
    ax.scatter(X[:, 0], X[:, 1], c=y_plot, cmap=cmap, norm=norm, s=10, edgecolor="k", linewidth=0.2)
    ax.contour(xx, yy, Z_plot, levels=[0.5], colors="k", linewidths=0.5, linestyles="--")

    textstr = f"$\\mathbf{{Recall}}$: {recall:.3f}\n$\\mathbf{{Precision}}$: {precision:.3f}\n$\\mathbf{{F1}}$: {f1:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

    ax.grid(which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="gray")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    
    if title:
        ax.set_title(f"{title} Classifier", fontweight="bold", fontsize=10)

    ax.set_xlabel(r"$x_1$", fontweight="bold")
    if index == 0:
        ax.set_ylabel(r"$x_2$", fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    if index == 2:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Minority Class',
                   markerfacecolor=class0_color, markersize=8, markeredgecolor='k', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w', label='Majority Class',
                   markerfacecolor=class1_color, markersize=8, markeredgecolor='k', markeredgewidth=0.5),
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.02, -0.35), ncol=2, frameon=True, edgecolor='black')

    letters = ["(a)", "(b)", "(c)", "(d)"]
    ax.text(0.5, -0.275, letters[index], transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')

    return {"model": model, "xx": xx, "yy": yy, "Z": Z, "ax": ax}

# ----------- Main Execution -----------

if __name__ == "__main__":
    X, y = generate_concentric_rings(
        n_rings=6, points_per_ring=1200, start_radius=0.8, imbalance_ratio=0.1,
        ring_width=0.4, gap=0.5, noise_std=0.25, random_state=1
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fig, axes = plt.subplots(1, 4, figsize=(12, 7), sharey=True)

    # MotherNet
    mn_model = MotherNetClassifier(device='cuda')
    plot_decision_regions(mn_model, X_train, y_train, X_test, y_test, is_mixture=False, ax=axes[0], title="MotherNet", fit=True, index=0)

    # Dual-MoH alpha=0.25
    moh_model_25 = Dual_MoH(m=6, overlap=0.15, verbose=False, alpha=0.25, minority_cluster='BalancedKMeansLSA', majority_cluster='BalancedKMeansLSA')
    plot_decision_regions(moh_model_25, X_train, y_train, X_test, y_test, is_mixture=True, ax=axes[1], title=r"Dual-MoH ($\mathbf{\alpha=0.25}$)", fit=True, index=1)

    # Dual-MoH alpha=0.5
    moh_model_50 = Dual_MoH(m=6, overlap=0.15, verbose=False, alpha=0.5, minority_cluster='BalancedKMeansLSA', majority_cluster='BalancedKMeansLSA')
    plot_decision_regions(moh_model_50, X_train, y_train, X_test, y_test, is_mixture=True, ax=axes[2], title=r"Dual-MoH ($\mathbf{\alpha=0.5}$)", fit=True, index=2)

    # Dual-MoH alpha=0.75
    moh_model_75 = Dual_MoH(m=6, overlap=0.15, verbose=False, alpha=0.75, minority_cluster='BalancedKMeansLSA', majority_cluster='BalancedKMeansLSA')
    plot_decision_regions(moh_model_75, X_train, y_train, X_test, y_test, is_mixture=True, ax=axes[3], title=r"Dual-MoH ($\mathbf{\alpha=0.75}$)", fit=True, index=3)

    fig.suptitle("Decision Boundaries on Concentric Rings Synthetic Dataset", fontweight='bold', y=0.70, fontsize=14)
    fig.subplots_adjust(wspace=0.1, bottom=0.02, right=0.95, left=0.05)
    
    #fig.savefig("decision_boundaries_concentric_rings_v9.png", dpi=800, bbox_inches='tight')
    plt.show()