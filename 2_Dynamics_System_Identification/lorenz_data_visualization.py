import numpy as np
import matplotlib.pyplot as plt

# Set global default parameters to improve plot appearance
plt.rcParams['figure.facecolor'] = 'white'  # Set the background color of the entire plot window to white
plt.rcParams['text.color'] = 'black'  # Set text color to black
plt.rcParams['axes.labelcolor'] = 'black'  # Set axis label color to black
plt.rcParams['xtick.color'] = 'black'  # Set tick color to black
plt.rcParams['ytick.color'] = 'black'  # Set tick color to black

# Load data
train_data = np.loadtxt('lorenz_dataset/train_features.txt')
test_data = np.loadtxt('lorenz_dataset/test_features.txt')
managed_data = np.loadtxt('lorenz_dataset/managed_features.txt')
sampled_data = np.loadtxt('lorenz_dataset/sampled_features.txt')

# Extract 3D coordinates
train_x, train_y, train_z = train_data[:, 0], train_data[:, 1], train_data[:, 2]
test_x, test_y, test_z = test_data[:, 0], test_data[:, 1], test_data[:, 2]
managed_x, managed_y, managed_z = managed_data[:, 0], managed_data[:, 1], managed_data[:, 2]
sampled_x, sampled_y, sampled_z = sampled_data[:, 0], sampled_data[:, 1], sampled_data[:, 2]

# Create a 1x3 subplot and adjust size and resolution
fig = plt.figure(figsize=(12, 4.8))

# Define point color, size, and transparency
colors = ['red', 'green', 'blue']
point_size = 2
point_alpha = 0.7  # Set point transparency to 0.7


# Create 3D scatter plots
for i, (data_x, data_y, data_z, title, data) in enumerate([
    (managed_x, managed_y, managed_z, r'$\mathcal{D}_{R}$', managed_data),
    (sampled_x, sampled_y, sampled_z, r'$\mathcal{D}_{M}$', sampled_data),
    (train_x, train_y, train_z, r'$\mathcal{D}$', train_data)
]):
    ax = fig.add_subplot(1, 3, 3 - i, projection='3d')

    # 使用黑色绘制点，确保与背景有良好的对比
    sc = ax.scatter(data_x, data_y, data_z, c=colors[i], s=point_size, alpha=point_alpha)

    # Calculate the number of samples and update the title
    num_samples = len(data)
    ax.set_title(f'|{title}|={num_samples}', fontsize=16, color='black')  # 添加样本数量到标题中

    ax.set_xlabel('x', fontsize=12, labelpad=0)
    ax.set_ylabel('y', fontsize=12, labelpad=0)
    ax.set_zlabel('z', fontsize=12, labelpad=0)

    # Set uniform axis limits to ensure consistent and compact plots
    all_data = np.vstack((train_data, test_data, managed_data, sampled_data))
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()
    margin = -0.01 * max(x_max - x_min, y_max - y_min, z_max - z_min)  # 添加5%的边界空间
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    ax.set_zlim([z_min - margin, z_max + margin])

    # Add axis ticks.
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax.view_init(elev=30, azim=-70)
    ax.grid(True)

# Improve layout and reduce spacing between subplots
plt.subplots_adjust(wspace=-0.2, left=0.0, right=1.0, hspace=0.0)
plt.savefig('lorenz_dataset.png')
plt.show()