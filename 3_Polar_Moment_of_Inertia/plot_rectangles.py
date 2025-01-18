import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define fixed display limits
fixed_xlim = (0, 280)
fixed_ylim = (0, 280)

plt.rcParams.update({
    'font.size': 14,        # Default font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # Axis label font size
    'legend.fontsize': 14,  # Legend font size
    'xtick.labelsize': 16,  # X-axis tick label font size
    'ytick.labelsize': 16,  # Y-axis tick label font size
})

# Function to load data from a file
def load_data(filename):
    """Load data from a specified file path."""
    return np.loadtxt(filename)

# Function to plot rectangles
def plot_rectangles(rect_data, I_data, axs):
    """Plot rectangles on the given axes."""
    for idx, rect in enumerate(rect_data):
        # Each rectangle data format: [bottom_left_x, bottom_left_y, top_right_x, top_right_y]
        x_bottom_left, y_bottom_left, x_top_right, y_top_right = rect

        # Calculate the width and height of the rectangle
        width = x_top_right - x_bottom_left
        height = y_top_right - y_bottom_left

        # Add a rectangle to the subplot
        rect_patch = patches.Rectangle((x_bottom_left, y_bottom_left), width, height,
                                       linewidth=1, edgecolor='white', facecolor='white')

        # Set the background color of each subplot to black
        axs[idx].add_patch(rect_patch)
        axs[idx].set_facecolor('black')

        xticks = np.arange(fixed_xlim[0], fixed_xlim[1] + 1, 40)
        yticks = np.arange(fixed_ylim[0], fixed_ylim[1] + 1, 40)
        axs[idx].set_xticks(xticks)
        axs[idx].set_yticks(yticks)

        # Set fixed axis limits to 280x280
        axs[idx].set_xlim(fixed_xlim)
        axs[idx].set_ylim(fixed_ylim)
        axs[idx].set_title(f'polar moment: {I_data[idx]:.3f}$\\times10^8$')

# Main program
if __name__ == "__main__":
    # Load data from files
    rect_features_filename = 'rectangles_dataset/train_features.txt'
    rect_labels_filename = 'rectangles_dataset/train_labels.txt'
    rect_data = load_data(rect_features_filename)
    I_data = load_data(rect_labels_filename)

    # Select the first three rectangles for plotting
    rect_data = rect_data[8:11]
    I_data = I_data[8:11]

    # Create a new figure
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Plot rectangles
    plot_rectangles(rect_data, I_data, axs)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()