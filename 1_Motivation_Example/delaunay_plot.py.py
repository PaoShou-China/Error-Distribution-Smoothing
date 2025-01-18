import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.gridspec import GridSpec

# 定义全局绘图参数
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
})


# Define the function f(x, y)
def f(x, y):
    """
    Compute the value of the function f(x, y) = 1 / (0.33 + x^2 + y^2).
    """
    return 1 / (0.33 + x ** 2 + y ** 2)


# Function to read data from a file
def read_data(filename):
    """Read data from a file and scale it by 0.6."""
    with open('dataset/' + filename, 'r') as file:
        data = np.loadtxt(file) * 0.6
    return data


def plot_with_delaunay(features, labels, title, subplot_position, shared_sm):
    """Plot the Delaunay triangulation with a contour filled background."""
    tri = Delaunay(features)

    # Calculate the color values for each triangle centroid
    barycenters = np.array([features[s].mean(axis=0) for s in tri.simplices])
    colors = [f(center[0], center[1]) for center in barycenters]

    ax = plt.subplot(subplot_position)
    ax.set_aspect('equal', 'box')

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)

    # Set uniform tick positions
    ticks = np.linspace(-3.0, 3.0, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    ax.contourf(X, Y, Z, levels=30, cmap='plasma')

    # Plot the Delaunay triangulation with white edges
    tripcolor_plot = ax.triplot(features[:, 0], features[:, 1], tri.simplices,
                                'go-', color='white', lw=0.3, ms=0.2)  # Set line width to 0.3 and point size to 2
    # scatter = ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='plasma', edgecolor='k', s=0.3)

    ax.set_title(title)

    return tripcolor_plot


# Read data
train_features = read_data('train_features.txt')
train_labels = read_data('train_labels.txt')
sampled_features = read_data('sampled_features.txt')
sampled_labels = read_data('sampled_labels.txt')
managed_features = read_data('managed_features.txt')
managed_labels = read_data('managed_labels.txt')

# Create the figure and plot three subplots
fig = plt.figure(figsize=(13, 4))

gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

# Collect all color values to determine the range for the shared colorbar
all_colors = []
for features in [train_features, sampled_features, managed_features]:
    tri = Delaunay(features)
    barycenters = np.array([features[s].mean(axis=0) for s in tri.simplices])
    colors = [f(center[0], center[1]) for center in barycenters]
    all_colors.extend(colors)

vmin = min(all_colors)
vmax = max(all_colors)

# Create a ScalarMappable object for the shared colorbar
shared_sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
shared_sm.set_array([])

# Plot the three subplots and hide y-axis ticks for the 2nd and 3rd subplots
plots = [
    plot_with_delaunay(train_features, train_labels, '$\mathcal{D}$', gs[0, 0], shared_sm),
    plot_with_delaunay(sampled_features, sampled_labels, '$\mathcal{D}_M$', gs[0, 1], shared_sm),
    plot_with_delaunay(managed_features, managed_labels, '$\mathcal{D}_R$', gs[0, 2], shared_sm)
]

# Hide y-axis tick labels and ticks for the 2nd and 3rd subplots
for i, ax in enumerate(fig.axes[1:3], start=1):
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position('none')

# Add the shared colorbar
cbar_ax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(shared_sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('$f\ (x, y)$', size=14)
cbar.ax.tick_params(labelsize=12)

# Minimize the spacing between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.01)

plt.show()
