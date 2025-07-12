import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import ScalarFormatter
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def get_uniform_data(n_points):
    def f(x, y):
        return 1 / (0.33 + x ** 2 + y ** 2)
    x = np.linspace(-3, 3, int(np.sqrt(n_points)) + 1)
    y = np.linspace(-3, 3, int(np.sqrt(n_points)) + 1)
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    return x.flatten(), y.flatten(), z.flatten()

class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.optimizer = optim.LBFGS(self.parameters(), lr=0.1, max_iter=30, tolerance_grad=1e-08,
                                     tolerance_change=1e-09, history_size=500)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, features):
        return self.net(features)

    def predict(self, x, y):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        features = torch.vstack([x, y]).T
        with torch.no_grad():
            label = self.net.forward(features)
            label = label.detach().cpu().numpy().flatten()
        return label

def read_data(filename):
    return np.loadtxt(filename)

def get_dataloader(features, labels):
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    return dataloader

def train(dataloader, epochs=200):
    net = Net(hidden_size=64).to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        def closure():
            net.optimizer.zero_grad()
            for features, labels in dataloader:
                predictions = net(features)
                loss = net.loss_fn(predictions, labels)
                loss.backward()
            return loss
        net.optimizer.step(closure)
    return net

def test(net, n_points, type_name, ax, vmin=None, vmax=None):
    x, y, z = get_uniform_data(n_points)
    z_hat = net.predict(x, y)
    delta = np.abs(z_hat - z) + 1e-5
    return np.min(delta), np.max(delta)

def main():
    datasets = [
        {"name": "train", "features": 'data/processed/1/train_features.txt', "labels": 'data/processed/1/train_labels.txt'},
        {"name": "sampled", "features": 'data/processed/1/sampled_features.txt', "labels": 'data/processed/1/sampled_labels.txt'},
        {"name": "smogn", "features": 'data/processed/1/smogn_features_dim0.txt', "labels": 'data/processed/1/smogn_labels_dim0.txt'},
        {"name": "managed", "features": 'data/processed/1/eds_features_dim0.txt', "labels": 'data/processed/1/eds_labels_dim0.txt'}
    ]
    nets = {}
    deltas = []
    # Create figure and GridSpec
    fig = plt.figure(figsize=(10, 9.3))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], 
                  wspace=0.2, hspace=0.10)
    for dataset in datasets:
        print(f"Processing dataset: {dataset['name']}")
        features = read_data(dataset["features"])
        labels = read_data(dataset["labels"])
        dataloader = get_dataloader(features, labels)
        net = train(dataloader)
        nets[dataset['name']] = net
    for i, dataset in enumerate(datasets):
        net = nets[dataset['name']]
        min_delta, max_delta = test(net, 1000 * 1000, dataset["name"], None)
        deltas.append((min_delta, max_delta))
    global_min = min(min(d) for d in deltas)
    global_max = max(max(d) for d in deltas)
    norm = LogNorm(vmin=global_min, vmax=global_max)
    cmap = plt.get_cmap('plasma')
    positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
    name_list = ['$\mathcal{D}$', '$\mathcal{D}_M$', '$\mathcal{D}_S$', '$\mathcal{D}_R$']
    # Draw four subplots
    for i, (dataset, pos) in enumerate(zip(datasets, positions)):
        net = nets[dataset['name']]
        x, y, z = get_uniform_data(1000 * 1000)
        z_hat = net.predict(x, y)
        delta = np.abs(z_hat - z) + 1e-5
        ax = fig.add_subplot(pos)
        sc = ax.scatter(x, y, c=delta, cmap=cmap, norm=norm, s=16)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ticks = np.linspace(-3.0, 3.0, 4)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{name_list[i]}\nRMSE:{round(np.mean(delta**2)**0.5, 5)}\n Max Error:{round(np.max(delta), 5)}")
        ax.set_xlabel('')
        ax.set_ylabel('')
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, 2])
    cbar = plt.colorbar(sc, cax=cbar_ax, label="Error")
    cbar.set_label('Error', size=14)
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0.05, hspace=0.10, top=0.95, bottom=0.05)
    plt.savefig("results/figures/1/4.png", dpi=300)
    # plt.show()

if __name__ == '__main__':
    main()