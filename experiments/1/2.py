import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.gridspec import GridSpec
import os

# Set global plotting parameters
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
})

def f(x, y):
    return 1 / (0.33 + x ** 2 + y ** 2)

def read_data(filename):
    with open(filename, 'r') as file:
        data = np.loadtxt(file)
    return data

def plot_with_delaunay(features, labels, title, subplot_position, shared_sm):
    tri = Delaunay(features)
    # Calculate color values for each triangle barycenter
    barycenters = np.array([features[s].mean(axis=0) for s in tri.simplices])
    colors = [f(center[0], center[1]) for center in barycenters]

    ax = plt.subplot(subplot_position)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ticks = np.linspace(-3.0, 3.0, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    x = np.linspace(-3, 3, 400)  # Reduce the number of points for better performance
    y = np.linspace(-3, 3, 400)  # Reduce the number of points for better performance
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    ax.contourf(X, Y, Z, levels=30, cmap='plasma')
    # Draw Delaunay triangulation with white edges
    tripcolor_plot = ax.triplot(features[:, 0], features[:, 1], tri.simplices,
                                'go-', color='white', lw=0.4, ms=0.6)
    ax.set_title(title)
    return tripcolor_plot

# Read data
train_features = read_data('data/processed/1/train_features.txt')
train_labels = read_data('data/processed/1/train_labels.txt')
sampled_features = read_data('data/processed/1/sampled_features.txt')
sampled_labels = read_data('data/processed/1/sampled_labels.txt')
managed_features = read_data('data/processed/1/eds_features_dim0.txt')
managed_labels = read_data('data/processed/1/eds_labels_dim0.txt')
smogn_features = read_data('data/processed/1/smogn_features_dim0.txt')
smogn_labels = read_data('data/processed/1/smogn_labels_dim0.txt')

# Create figure and draw 2x2 subplots
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], 
              wspace=0.2, hspace=0.15)

# Collect all color values for shared colorbar range
all_colors = []
for features in [train_features, sampled_features, managed_features, smogn_features]:
    tri = Delaunay(features)
    barycenters = np.array([features[s].mean(axis=0) for s in tri.simplices])
    colors = [f(center[0], center[1]) for center in barycenters]
    all_colors.extend(colors)

vmin = min(all_colors)
vmax = max(all_colors)

# Create ScalarMappable for shared colorbar
shared_sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
shared_sm.set_array([])

# Draw 2x2 subplots
plots = [
    plot_with_delaunay(train_features, train_labels, '$|\mathcal{D}|=5000$', gs[0, 0], shared_sm),
    plot_with_delaunay(sampled_features, sampled_labels, '$|\mathcal{D}_M|=648$', gs[0, 1], shared_sm),
    plot_with_delaunay(smogn_features, smogn_labels, '$|\mathcal{D}_S|=4038$', gs[1, 0], shared_sm),
    plot_with_delaunay(managed_features, managed_labels, '$|\mathcal{D}_R|=648$', gs[1, 1], shared_sm)
]
# Set tick label size for all axes
for i, ax in enumerate(fig.axes):
    ax.tick_params(axis='both', which='both', labelsize=12)

# Add shared colorbar
cbar_ax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(shared_sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('$f\ (x, y)$', size=14)
cbar.ax.tick_params(labelsize=12)

plt.subplots_adjust(wspace=0.05, hspace=0.15, top=0.95, bottom=0.05)
plt.savefig('results/figures/1/2.png', dpi=300)
plt.show()