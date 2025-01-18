import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from scipy.spatial import Delaunay
from scipy.linalg import norm

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,  # Global font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label font size
    'ytick.labelsize': 14,  # Y-axis tick label font size
    'legend.fontsize': 16,  # Legend font size
})


# Define the function f(x, y)
def f(x, y):
    return 1 / (0.33 + x ** 2 + y ** 2)


# Compute the Hessian matrix
def compute_hessian_matrix(x, y):
    """
    Compute the Hessian matrix of the function f(x, y).
    """
    denominator = (0.33 + x ** 2 + y ** 2) ** 3
    hxx = (8 * x ** 2 - 2 * (0.33 + x ** 2 + y ** 2)) / denominator
    hyy = (8 * y ** 2 - 2 * (0.33 + x ** 2 + y ** 2)) / denominator
    hxy = hyx = (-4 * x * y) / denominator
    return np.array([[hxx, hxy], [hyx, hyy]])


# Function to read data from a file
def read_data(filename):
    with open('dataset/' + filename, 'r') as file:
        data = np.loadtxt(file)
    return data


# Compute the Hessian F-norms for the vertices
def compute_hessian_norms(features):
    """
    Compute the Hessian F-norms for the vertices of the Delaunay triangulation.
    """
    tri = Delaunay(features)

    # Compute the Hessian F-norm for each simplex
    vertex_hessian_norms = np.zeros(features.shape[0])
    for simplex in tri.simplices:
        vertices = features[simplex]
        max_edge_length = max(np.linalg.norm(vertices[i] - vertices[j]) for i in range(3) for j in range(i + 1, 3))
        center = vertices.mean(axis=0)
        hessian = compute_hessian_matrix(center[0], center[1])
        hessian_f_norm = norm(hessian, 'fro') * (max_edge_length ** 2)

        for vertex_index in simplex:
            vertex_hessian_norms[vertex_index] += hessian_f_norm

    # Average the F-norms for each vertex( Because a vertex may belong to multiple simplex)
    unique_vertex_counts = np.bincount(tri.simplices.ravel(), minlength=len(features))
    vertex_hessian_norms /= unique_vertex_counts

    return vertex_hessian_norms


# Plot scatter plot and color vertices by log Hessian F-norm
def plot_scatter(features, title, n_samples, subplot_position, vmin, vmax, cmap, axis_limits, tick_interval,
                 show_ylabel=False):
    """
    Plot a scatter plot and color vertices by the log Hessian F-norm.
    """
    log_vertex_hessian_norms = compute_hessian_norms(features)

    plt.subplot(subplot_position)

    scatter = plt.scatter(
        features[:, 0] * 0.6,
        features[:, 1] * 0.6,
        c=log_vertex_hessian_norms,
        cmap=cmap,
        s=16,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    plt.title(f'|{title}|={n_samples}', fontsize=16)
    plt.xlabel('x', fontsize=14)

    # Set x-axis and y-axis limits and tick intervals
    plt.xlim(axis_limits)
    plt.ylim(axis_limits)
    plt.xticks(np.arange(axis_limits[0], axis_limits[1] + tick_interval, tick_interval))
    plt.yticks(np.arange(axis_limits[0], axis_limits[1] + tick_interval, tick_interval))

    if show_ylabel:
        plt.ylabel('y', fontsize=14)  # Show y-axis label only for the leftmost subplot
    else:
        plt.ylabel('')
        plt.yticks([])  # Hide y-axis ticks for other subplots

    plt.axis('equal')
    return scatter


# Read data
train_features = read_data('train_features.txt')
sampled_features = read_data('sampled_features.txt')
managed_features = read_data('managed_features.txt')

# Get the number of samples in each dataset
n_train_samples = train_features.shape[0]
n_sampled_samples = sampled_features.shape[0]
n_managed_samples = managed_features.shape[0]

# Compute global minimum and maximum log Hessian F-norms
all_log_norms = np.concatenate([
    compute_hessian_norms(train_features),
    compute_hessian_norms(sampled_features),
    compute_hessian_norms(managed_features)
])
vmin, vmax = np.min(all_log_norms), np.max(all_log_norms)

cmap = plt.get_cmap('plasma')

fig = plt.figure(figsize=(12, 4))


# Compute the minimum and maximum values of all features to determine the uniform axis range
all_features = np.vstack([train_features, sampled_features, managed_features]) * 0.6
min_val = np.min(all_features, axis=0)
max_val = np.max(all_features, axis=0)
axis_range = min_val[0], max_val[0]  # Assume x and y have the same range
tick_interval = (max_val[0] - min_val[0]) / 3

# Plot the three subplots and show the number of samples in the title
scatter1 = plot_scatter(train_features, '$\mathcal{D}$', n_train_samples, 131, vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=True)
scatter2 = plot_scatter(sampled_features, '$\mathcal{D}_{M}$', n_sampled_samples, 132, vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=False)
scatter3 = plot_scatter(managed_features, '$\mathcal{D}_{R}$', n_managed_samples, 133, vmin, vmax, cmap, axis_range,
                        tick_interval, show_ylabel=False)

# 添加全局颜色条，设置为对数刻度，并设置字体大小
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter3, cax=cax, label='CDR', extend='both')
cbar.set_label('CDR', size=14)  # 设置颜色条标签字体大小
cbar.ax.tick_params(labelsize=12)  # 设置颜色条刻度标签字体大小

# Minimize the spacing between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.01, bottom=0.15)  # Set small width and height spacing

plt.show()
