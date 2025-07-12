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
train_data = np.loadtxt('data/processed/2/train_features.txt')
test_data = np.loadtxt('data/processed/2/test_features.txt')
# sampled_data = np.loadtxt('data/processed/2/sampled_features.txt')
managed_data_dim0 = np.loadtxt('data/processed/2/eds_features_dim0.txt')
managed_data_dim1 = np.loadtxt('data/processed/2/eds_features_dim1.txt')
managed_data_dim2 = np.loadtxt('data/processed/2/eds_features_dim2.txt')
smogn_data_dim0 = np.loadtxt('data/processed/2/smogn_features_dim0.txt')
smogn_data_dim1 = np.loadtxt('data/processed/2/smogn_features_dim1.txt')
smogn_data_dim2 = np.loadtxt('data/processed/2/smogn_features_dim2.txt')

# Extract 3D coordinates
train_x, train_y, train_z = train_data[:, 0], train_data[:, 1], train_data[:, 2]
test_x, test_y, test_z = test_data[:, 0], test_data[:, 1], test_data[:, 2]
managed_x_dim0, managed_y_dim0, managed_z_dim0 = managed_data_dim0[:, 0], managed_data_dim0[:, 1], managed_data_dim0[:, 2]
managed_x_dim1, managed_y_dim1, managed_z_dim1 = managed_data_dim1[:, 0], managed_data_dim1[:, 1], managed_data_dim1[:, 2]
managed_x_dim2, managed_y_dim2, managed_z_dim2 = managed_data_dim2[:, 0], managed_data_dim2[:, 1], managed_data_dim2[:, 2]
smogn_x_dim0, smogn_y_dim0, smogn_z_dim0 = smogn_data_dim0[:, 0], smogn_data_dim0[:, 1], smogn_data_dim0[:, 2]
smogn_x_dim1, smogn_y_dim1, smogn_z_dim1 = smogn_data_dim1[:, 0], smogn_data_dim1[:, 1], smogn_data_dim1[:, 2]
smogn_x_dim2, smogn_y_dim2, smogn_z_dim2 = smogn_data_dim2[:, 0], smogn_data_dim2[:, 1], smogn_data_dim2[:, 2]
# sampled_x, sampled_y, sampled_z = sampled_data[:, 0], sampled_data[:, 1], sampled_data[:, 2]

# Randomly sample from train_data to match the number of managed_data_dim0
np.random.seed(42)  # Fix random seed for reproducibility
sampled_indices = np.random.choice(train_data.shape[0], managed_data_dim0.shape[0], replace=False)
sampled_random_data = train_data[sampled_indices]
sampled_random_x, sampled_random_y, sampled_random_z = sampled_random_data[:, 0], sampled_random_data[:, 1], sampled_random_data[:, 2]

# Create 2x2 subplots and adjust size and resolution
fig = plt.figure(figsize=(9, 9))

# Define point colors, size, and transparency
colors = ['red', 'green', 'blue', 'black']
point_size = 2
point_alpha = 0.7  # Set point transparency to 0.7

# Get coordinate ranges for all data
all_data = np.vstack((train_data, test_data, managed_data_dim0, smogn_data_dim0))
x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()
margin = -0.01 * max(x_max - x_min, y_max - y_min, z_max - z_min)

# Draw 3D scatter plots
for i, (data_x, data_y, data_z, title, data) in enumerate([
    (managed_x_dim0, managed_y_dim0, managed_z_dim0, r'$\mathcal{D}_{R}$', managed_data_dim0),
    (sampled_random_x, sampled_random_y, sampled_random_z, r'$\mathcal{D}_{M}$', sampled_random_data),
    (smogn_x_dim0, smogn_y_dim0, smogn_z_dim0, r'$\mathcal{D}_{S}$', smogn_data_dim0),
    (train_x, train_y, train_z, r'$\mathcal{D}$', train_data)
]):
    # Calculate subplot position
    row = i // 2
    col = i % 2
    ax = fig.add_subplot(2, 2, 4-i, projection='3d')
    sc = ax.scatter(data_x, data_y, data_z, c=colors[i], s=point_size, alpha=point_alpha)
    num_samples = len(data)
    ax.set_title(f'|{title}|={num_samples}', fontsize=16, color='black')
    ax.set_xlabel('x', fontsize=12, labelpad=0)
    ax.set_ylabel('y', fontsize=12, labelpad=0)
    ax.set_zlabel('z', fontsize=12, labelpad=0)
    # Set unified axis ranges
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    ax.set_zlim([z_min - margin, z_max + margin])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.view_init(elev=30, azim=-70)
    ax.grid(True)

# Adjust layout
plt.subplots_adjust(wspace=0.05, hspace=0.2, left=0.02, right=0.98, bottom=0.02, top=0.95)
plt.savefig('results/figures/2/1.png', dpi=300)
plt.show()