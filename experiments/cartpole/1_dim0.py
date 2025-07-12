import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting functionality
import os

# Set global default parameters for better plot appearance
plt.rcParams['figure.facecolor'] = 'white'  # Set background color to white
plt.rcParams['text.color'] = 'black'       # Set text color to black
plt.rcParams['axes.labelcolor'] = 'black'  # Set axis label color to black
plt.rcParams['xtick.color'] = 'black'      # Set x-tick color to black
plt.rcParams['ytick.color'] = 'black'      # Set y-tick color to black

# Load data
train_data = np.loadtxt('data/processed/3/train_features.txt')
test_data = np.loadtxt('data/processed/3/test_features.txt')
eds_dim0_data = np.loadtxt('data/processed/3/eds_features_dim0.txt')
eds_dim1_data = np.loadtxt('data/processed/3/eds_features_dim1.txt')
smogn_dim0_data = np.loadtxt('data/processed/3/smogn_features_dim0.txt')
smogn_dim1_data = np.loadtxt('data/processed/3/smogn_features_dim1.txt')

# Randomly sample from train_data to match the number of EDS samples for each dim
np.random.seed(42)  # Fix random seed for reproducibility
sampled_dim0_indices = np.random.choice(train_data.shape[0], eds_dim0_data.shape[0], replace=False)
sampled_dim1_indices = np.random.choice(train_data.shape[0], eds_dim1_data.shape[0], replace=False)
sampled_dim0_data = train_data[sampled_dim0_indices]
sampled_dim1_data = train_data[sampled_dim1_indices]

# Extract 3D coordinates
train_x, train_y, train_z = train_data[:, 0], train_data[:, 1], train_data[:, 2]
test_x, test_y, test_z = test_data[:, 0], test_data[:, 1], test_data[:, 2]
eds_dim0_x, eds_dim0_y, eds_dim0_z = eds_dim0_data[:, 0], eds_dim0_data[:, 1], eds_dim0_data[:, 2]
eds_dim1_x, eds_dim1_y, eds_dim1_z = eds_dim1_data[:, 0], eds_dim1_data[:, 1], eds_dim1_data[:, 2]
smogn_dim0_x, smogn_dim0_y, smogn_dim0_z = smogn_dim0_data[:, 0], smogn_dim0_data[:, 1], smogn_dim0_data[:, 2]
smogn_dim1_x, smogn_dim1_y, smogn_dim1_z = smogn_dim1_data[:, 0], smogn_dim1_data[:, 1], smogn_dim1_data[:, 2]
sampled_dim0_x, sampled_dim0_y, sampled_dim0_z = sampled_dim0_data[:, 0], sampled_dim0_data[:, 1], sampled_dim0_data[:, 2]
sampled_dim1_x, sampled_dim1_y, sampled_dim1_z = sampled_dim1_data[:, 0], sampled_dim1_data[:, 1], sampled_dim1_data[:, 2]

# Create 1x5 subplots and adjust size and resolution
fig = plt.figure(figsize=(9, 9))

# Define point colors, size, and transparency
colors = ['red', 'green', 'blue', 'black']
point_size = 2
point_alpha = 0.7  # Set point transparency to 0.7

# Draw 3D scatter plots
for i, (data_x, data_y, data_z, title, data) in enumerate([
    (eds_dim0_x, eds_dim0_y, eds_dim0_z, r'$\mathcal{D}_{R}$', eds_dim0_data),
    (sampled_dim0_x, sampled_dim0_y, sampled_dim0_z, r'$\mathcal{D}_{M}$', sampled_dim0_data),
    (smogn_dim0_x, smogn_dim0_y, smogn_dim0_z, r'$\mathcal{D}_{S}$', smogn_dim0_data),
    (train_x, train_y, train_z, r'$\mathcal{D}$', train_data)
]):
    ax = fig.add_subplot(2, 2, 4 - i, projection='3d')
    # Use black color for points to ensure good contrast with background
    sc = ax.scatter(data_x, data_y, data_z, c=colors[i], s=point_size, alpha=point_alpha)
    num_samples = len(data)
    ax.set_title(f'|{title}|={num_samples}', fontsize=16, color='black')
    ax.set_xlabel('$\\theta$', fontsize=12, labelpad=0)
    ax.set_ylabel('$\omega$', fontsize=12, labelpad=0)
    ax.set_zlabel('$I$', fontsize=12, labelpad=0)
    # Set unified axis ranges for consistent plot layout
    all_data = np.vstack((train_data, test_data, eds_dim0_data, eds_dim1_data, smogn_dim0_data, smogn_dim1_data))
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()
    margin = -0.01 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    ax.set_zlim([z_min - margin, z_max + margin])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.view_init(elev=60, azim=200)
    ax.grid(True)

# Adjust layout
plt.subplots_adjust(wspace=0.05, hspace=0.2, left=0.02, right=0.98, bottom=0.02, top=0.95)
plt.savefig('results/figures/3/1_dim0.png', dpi=300)
plt.show()