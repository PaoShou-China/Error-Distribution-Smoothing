import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from scipy.spatial import Delaunay
from scipy.linalg import norm
import os
from matplotlib.gridspec import GridSpec

# Set global font sizes for plots
plt.rcParams.update({
    'font.size': 14,  # Global font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label font size
    'ytick.labelsize': 14,  # Y-axis tick label font size
    'legend.fontsize': 16,  # Legend font size
})

def f(x, y):
    return 1 / (0.33 + x ** 2 + y ** 2)

def hessian_f(x, y):
    denominator = (0.33 + x ** 2 + y ** 2) ** 3
    hxx = (8 * x ** 2 - 2 * (0.33 + x ** 2 + y ** 2)) / denominator
    hyy = (8 * y ** 2 - 2 * (0.33 + x ** 2 + y ** 2)) / denominator
    hxy = hyx = (-4 * x * y) / denominator
    return np.array([[hxx, hxy], [hyx, hyy]])

def read_data(filename):
    with open(filename, 'r') as file:
        data = np.loadtxt(file)
    return data

def compute_hessian_norms(features):
    tri = Delaunay(features)
    # Compute Frobenius norm of Hessian for each triangle
    vertex_hessian_norms = np.zeros(features.shape[0])
    for simplex in tri.simplices:
        vertices = features[simplex]
        max_edge_length = max(np.linalg.norm(vertices[i] - vertices[j]) for i in range(3) for j in range(i + 1, 3))
        center = vertices.mean(axis=0)
        hessian = hessian_f(center[0], center[1])
        hessian_f_norm = norm(hessian, 'fro') * (max_edge_length ** 2)
        for vertex_index in simplex:
            vertex_hessian_norms[vertex_index] += hessian_f_norm
    # Average the Frobenius norm for each vertex
    unique_vertex_counts = np.bincount(tri.simplices.ravel(), minlength=len(features))
    vertex_hessian_norms /= unique_vertex_counts
    return vertex_hessian_norms

def plot_scatter(features, title, n_samples, subplot_position, vmin, vmax, cmap, axis_limits, tick_interval,
                 show_ylabel=False):
    log_vertex_hessian_norms = compute_hessian_norms(features)
    plt.subplot(subplot_position)
    scatter = plt.scatter(
        features[:, 0],
        features[:, 1],
        c=log_vertex_hessian_norms,
        cmap=cmap,
        s=16,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    plt.title(f'|{title}|={n_samples}', fontsize=16)
    plt.xlabel('x', fontsize=14)
    # Set axis range and tick intervals
    plt.xlim(axis_limits)
    plt.ylim(axis_limits)
    plt.xticks(np.arange(axis_limits[0], axis_limits[1] + tick_interval, tick_interval))
    plt.yticks(np.arange(axis_limits[0], axis_limits[1] + tick_interval, tick_interval))
    # Hide axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('equal')
    return scatter

# Read data
train_features = read_data('data/processed/1/train_features.txt')
train_labels = read_data('data/processed/1/train_labels.txt')
sampled_features = read_data('data/processed/1/sampled_features.txt')
sampled_labels = read_data('data/processed/1/sampled_labels.txt')
managed_features = read_data('data/processed/1/eds_features_dim0.txt')
managed_labels = read_data('data/processed/1/eds_labels_dim0.txt')
smogn_features = read_data('data/processed/1/smogn_features_dim0.txt')
smogn_labels = read_data('data/processed/1/smogn_labels_dim0.txt')

# Get sample counts for each dataset
n_train_samples = train_features.shape[0]
n_sampled_samples = sampled_features.shape[0]
n_managed_samples = managed_features.shape[0]
n_smogn_samples = smogn_features.shape[0]

# Compute global min and max for Frobenius norm values
all_log_norms = np.concatenate([
    compute_hessian_norms(train_features),
    compute_hessian_norms(sampled_features),
    compute_hessian_norms(managed_features),
    compute_hessian_norms(smogn_features)
])
vmin, vmax = np.min(all_log_norms), np.max(all_log_norms)

cmap = plt.get_cmap('plasma')

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], 
              wspace=0.2, hspace=0.16)

# Compute axis range and tick interval
all_features = np.vstack([train_features, sampled_features, managed_features, smogn_features])
min_val = np.min(all_features, axis=0)
max_val = np.max(all_features, axis=0)
axis_range = min_val[0], max_val[0]
tick_interval = (max_val[0] - min_val[0]) / 3

# Plot four subplots using the defined function
scatter1 = plot_scatter(train_features, '$\mathcal{D}$', n_train_samples, gs[0, 0], vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=True)
scatter2 = plot_scatter(sampled_features, '$\mathcal{D}_M$', n_sampled_samples, gs[0, 1], vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=False)
scatter3 = plot_scatter(smogn_features, '$\mathcal{D}_S$', n_smogn_samples, gs[1, 0], vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=True)
scatter4 = plot_scatter(managed_features, '$\mathcal{D}_R$', n_managed_samples, gs[1, 1], vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=False)

# Add global colorbar
cbar_ax = fig.add_subplot(gs[:, 2])
cbar = plt.colorbar(scatter4, cax=cbar_ax, label='CDR', extend='both')
cbar.set_label('CDR', size=14)
cbar.ax.tick_params(labelsize=12)

plt.subplots_adjust(wspace=0.05, hspace=0.16, top=0.95, bottom=0.05)
plt.savefig('results/figures/1/3.png', dpi=300)
plt.show()