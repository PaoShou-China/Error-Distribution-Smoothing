import numpy as np
import matplotlib.pyplot as plt


# Function to load low-dimensional features from a file
def load_low_dim_features(features_path):
    """
    Load low-dimensional features from a specified file path.
    """
    try:
        return np.loadtxt('dataset/' + features_path)
    except Exception as e:
        print(f"Error loading {features_path}: {e}")
        return None


# Function to plot the centers of rectangles
def plot_rectangle_centers(low_dim_features, ax, label, color, name, marker='o', size=15, alpha=0.7, edgecolors='w', linewidth=0.5):
    """
    Plot the centers of rectangles on the given axis.
    """
    if low_dim_features is None or len(low_dim_features) == 0:
        print(f"No data for {label}.")
        return

    # Calculate the center coordinates of each rectangle
    centers_x = (low_dim_features[:, 0] + low_dim_features[:, 2]) / 2
    centers_y = (low_dim_features[:, 1] + low_dim_features[:, 3]) / 2

    # Plot all center points using scatter
    scatter = ax.scatter(centers_x, centers_y,
                         color=color,
                         label=label,
                         marker=marker,
                         s=size,
                         alpha=alpha,
                         edgecolors=edgecolors,
                         linewidth=linewidth)

    # Set x and y axis labels with increased font size
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)

    # Add sample count to the subplot title
    sample_count = len(low_dim_features)
    ax.set_title(f'|{name}|={sample_count}', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)

    # Adjust tick label font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Set axis limits to ensure all points are visible
    if sample_count > 0:
        min_x = np.min(low_dim_features[:, 0])
        max_x = np.max(low_dim_features[:, 2])
        min_y = np.min(low_dim_features[:, 1])
        max_y = np.max(low_dim_features[:, 3])
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)

    return scatter


if __name__ == "__main__":
    datasets = ['train', 'sampled', 'managed']
    colors = ['blue', 'green', 'red']
    names = ['$\mathcal{D}$', '$\mathcal{D}_M$', '$\mathcal{D}_R$']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))  # Create a 1x3 subplot

    scatters = []
    all_x = []
    all_y = []

    # First, iterate through all datasets to determine the global x and y axis ranges
    for head in datasets:
        low_dim_features = load_low_dim_features(f'{head}_features.txt')
        if low_dim_features is not None and len(low_dim_features) > 0:
            all_x.extend([low_dim_features[:, 0], low_dim_features[:, 2]])
            all_y.extend([low_dim_features[:, 1], low_dim_features[:, 3]])

    if all_x and all_y:  # 确保有数据存在
        min_x = min([min(x) for x in all_x]) +10
        max_x = max([max(x) for x in all_x]) -10
        min_y = min([min(y) for y in all_y]) +10
        max_y = max([max(y) for y in all_y]) -10

        # Define tick intervals
        tick_interval = 50
        x_ticks = np.arange(np.floor(min_x), np.ceil(max_x)+1, tick_interval)
        y_ticks = np.arange(np.floor(min_y), np.ceil(max_y)+1, tick_interval)

        # Plot the graphs and apply uniform axis ranges and ticks
        for idx, head in enumerate(datasets):
            low_dim_features = load_low_dim_features(f'{head}_features.txt')

            # Output some statistical information about the data for debugging
            if low_dim_features is not None and len(low_dim_features) > 0:
                print(f"Loaded {len(low_dim_features)} rectangles for {head} dataset.")
                print(f"First few entries: \n{low_dim_features[:5]}")  # 打印前几个条目

            scatter = plot_rectangle_centers(low_dim_features, axes[idx], head, colors[idx], names[idx])
            scatters.append(scatter)

            # Set uniform axis ranges and ticks
            axes[idx].set_aspect('equal', adjustable='box')
            axes[idx].set_xlim(min_x, max_x)
            axes[idx].set_ylim(min_y, max_y)
            axes[idx].set_xticks(x_ticks)
            axes[idx].set_yticks(y_ticks)

    # Add a common legend to the plot
    fig.legend(scatters, [s.get_label() for s in scatters], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Automatically adjust subplot parameters to fill the entire image area, leaving space for the legend.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()