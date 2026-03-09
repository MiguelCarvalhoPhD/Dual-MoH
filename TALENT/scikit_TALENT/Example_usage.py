import numpy as np
import matplotlib.pyplot as plt
from mixture_hypernetworks import *

#----------- Data Generation Function -----------

def generate_concentric_rings(n_rings=6,
                              points_per_ring=200,
                              start_radius=1.0,
                              ring_width=0.4,
                              gap=0.6,
                              noise_std=0.05,
                              random_state=None,
                              imbalance_ratio=0.2,
                              majority_class=0):
    """
    Generate 2D points arranged in concentric rings of alternating classes.

    New parameters:
    -----------
    imbalance_ratio : float
        If > 1.0, rings whose label == majority_class will have
        approximately points_per_ring * imbalance_ratio samples (rounded).
        If == 1.0, the dataset is balanced (no change).
    majority_class : int (0 or 1)
        Which class should be treated as the majority when imbalance_ratio > 1.
        Labels alternate per ring as (i % 2).
    """
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.RandomState(int(random_state))
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random

    # Normalize imbalance_ratio
    if imbalance_ratio <= 0:
        raise ValueError("imbalance_ratio must be positive.")

    # Determine base counts per ring
    if np.isscalar(points_per_ring):
        base_counts = [int(points_per_ring)] * n_rings
    else:
        if len(points_per_ring) != n_rings:
            raise ValueError("If points_per_ring is a sequence, its length must equal n_rings.")
        base_counts = [int(c) for c in points_per_ring]

    # Apply imbalance: scale counts of rings that belong to the majority class
    counts = []
    for i in range(n_rings):
        label = i % 2
        base_n = base_counts[i]
        if imbalance_ratio == 1.0:
            n = base_n
        else:
            if label == majority_class:
                # majority rings get scaled up
                n = int(round(base_n * imbalance_ratio))
            else:
                # minority rings keep base count
                n = int(round(base_n))
        # ensure at least 1 sample per ring
        n = max(1, n)
        counts.append(n)

    X_list = []
    y_list = []

    for i in range(n_rings):
        center_radius = start_radius + i * (ring_width + gap)
        n = counts[i]
        angles = rng.uniform(0, 2 * np.pi, size=n)
        radial_offsets = rng.uniform(-ring_width/2, ring_width/2, size=n)
        radii = center_radius + radial_offsets
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        xs += rng.normal(scale=noise_std, size=n)
        ys += rng.normal(scale=noise_std, size=n)
        X_ring = np.column_stack([xs, ys])
        label = i % 2  # alternate classes 0,1,0,1,...
        y_ring = np.full(n, label, dtype=int)
        X_list.append(X_ring)
        y_list.append(y_ring)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y



from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from sklearn.metrics import recall_score, precision_score, f1_score


def plot_decision_regions(
    model, X, y, X_test,y_test,
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
    """
    Train `model` on (X, y), create a 2D grid covering the data domain and plot
    predicted class regions, decision boundary, and training points.

    Added:
    - class0_color, class1_color: colors used for BOTH points and regions.
    - decision boundary plotted in black.

    Notes:
    - Assumes binary classification (2 classes).
    - Robust to arbitrary label values (e.g., {-1, 1}) by remapping to {0,1}.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[1] != 2:
        raise ValueError("X must be 2-dimensional (n_samples, 2).")

    # Fit model
    if fit and not is_mixture:
        model.fit(X, y)

    elif fit and is_mixture:
        model.fit_mixture_hypernetworks(X, y)


    # Grid bounds
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    print(X_test.shape, y_test.shape,X_train.shape, y_train.shape)

    # Predict on grid
    if not is_mixture:
        Z = model.predict(grid)
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

    else:
        _, _, Z, _, _, _, _ = model.predict_2(grid)
        _,_,y_pred,_,_,_,_ = model.predict_2(X_test)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
    Z = Z.reshape(xx.shape)

    print(f"{title} -- Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1:.4f}")

    # Create axes if needed
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        created_ax = True

    # ----- Binary class handling + colors -----
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"This version expects exactly 2 classes, but got {len(classes)}: {classes}"
        )

    # Remap labels -> {0,1} to make coloring consistent regardless of original labels
    y_plot = np.searchsorted(classes, y)
    Z_plot = np.searchsorted(classes, Z)

    # Colormap for BOTH regions + points
    class_colors = [class0_color, class1_color]
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)

    # ----- Regions -----
    mesh = ax.pcolormesh(
        xx, yy, Z_plot,
        shading="auto",
        alpha=0.3,
        cmap=cmap,
        norm=norm
    )

    # ----- Points -----
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y_plot,
        cmap=cmap,
        norm=norm,
        s=10,
        edgecolor="k",
        linewidth=0.2
    )

    # ----- Decision boundary (black) -----
    # boundary between class 0 and class 1 in Z_plot occurs at 0.5
    ax.contour(
        xx, yy, Z_plot,
        levels=[0.5],
        colors="k",
        linewidths=0.5,
        linestyles="--"
    )
    
    # add text box with the scores (only f1 and recall) in the upper left corner of the plot
    textstr = (
    f"$\\mathbf{{Recall}}$: {recall:.3f}\n"
    f"$\\mathbf{{Precision}}$: {precision:.3f}\n"
    f"$\\mathbf{{F1}}$: {f1:.3f}")

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

    # Styling
    ax.grid(which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="gray")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    if title is not None:
        ax.set_title(f"{title} Classifier", fontweight="bold",fontsize=10)

    if index == 0:
        ax.set_xlabel(r"$x_1$", fontweight="bold")
        ax.set_ylabel(r"$x_2$", fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlabel(r"$x_1$", fontweight="bold")

    # Legend matching the chosen class colors
    if index == 2:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Minority Class',
                   markerfacecolor=class0_color, markersize=8,
                   markeredgecolor='k', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w', label='Majority Class',
                   markerfacecolor=class1_color, markersize=8,
                   markeredgecolor='k', markeredgewidth=0.5),
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.02, -0.35),
            ncol=2,
            frameon=True,
            edgecolor='black'
        )
        
    letters = ["(a)", "(b)", "(c)", "(d)"]
    #add bolded text below each subplot
    ax.text(0.5, -0.275, letters[index], transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')

    if created_ax:
        print("Displaying plot...")

    return {"model": model, "xx": xx, "yy": yy, "Z": Z, "mesh": mesh, "ax": ax}


from sklearn.model_selection import train_test_split

#create dataset
X, y = generate_concentric_rings(n_rings=6, points_per_ring=1200, start_radius=0.8,imbalance_ratio=0.1,
                                 ring_width=0.4, gap=0.5, noise_std=0.25, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset shape:", X_train.shape, X_test.shape)

fig, axes = plt.subplots(1,4,figsize=(12,7),sharey=True)

# mothernet
mn_model = MotherNetClassifier(device='cuda')
plot_decision_regions(mn_model, X_train, y_train, X_test, y_test, is_mixture=False, ax=axes[0], title="MotherNet", fit=True)

# xgboost
# weights = compute_sample_weight(class_weight='balanced', y=y_train)
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train, sample_weight=weights)
# plot_decision_regions(xgb_model, X_train, y_train, X_test, y_test,is_mixture=False, ax=axes[3], title="CS XGBoost", fit=False,index=3)

# tabpfn
# tabpfn_model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
# plot_decision_regions(tabpfn_model, X, y, is_mixture=False, ax=axes[2], title="TabPFN", fit=True)

# MoH
moh_model = Dual_MoH(m=6,overlap=0.15,verbose=False,alpha=0.25,minority_cluster='BalancedKMeansLSA',majority_cluster='BalancedKMeansLSA')
plot_decision_regions(moh_model, X_train, y_train, X_test, y_test, is_mixture=True, ax=axes[1], title=r"Dual-MoH ($\mathbf{\alpha=0.25}$)", fit=True,index=1)

# custom ensemble
moh_model = Dual_MoH(m=6,overlap=0.15,verbose=False,alpha=0.5,minority_cluster='BalancedKMeansLSA',majority_cluster='BalancedKMeansLSA')
plot_decision_regions(moh_model, X_train, y_train, X_test, y_test, is_mixture=True, ax=axes[2], title=r"Dual-MoH ($\mathbf{\alpha=0.5}$)", fit=True,index=2)

moh_model = Dual_MoH(m=6,overlap=0.15,verbose=False,alpha=0.75,minority_cluster='BalancedKMeansLSA',majority_cluster='BalancedKMeansLSA')
plot_decision_regions(moh_model, X_train, y_train, X_test, y_test, is_mixture=True, ax=axes[3], title=r"Dual-MoH ($\mathbf{\alpha=0.75}$)", fit=True,index=3)


#add custom legend below the plots
#fig.tight_layout()

#define title of the entire figure
fig.suptitle("Decision Boundaries on Concentric Rings Synthetic Dataset",fontweight='bold', y=0.70, fontsize=14)

fig.subplots_adjust(wspace=0.1, bottom=0.02,right=0.95,left=0.05)
#fig.tight_layout()
#fig.savefig("decision_boundaries_concentric_rings_v9.png",dpi=800,bbox_inches='tight')
plt.show()
