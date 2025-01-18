import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import ScalarFormatter

# Set font sizes
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
    """
    Generate uniform data points and compute the function values.
    """
    def f(x, y):
        return 1 / (0.33 + x ** 2 + y ** 2)

    x = np.linspace(-3, 3, int(np.sqrt(n_points)) + 1)
    y = np.linspace(-3, 3, int(np.sqrt(n_points)) + 1)
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    return x.flatten(), y.flatten(), z.flatten()


class NeuralNetwork(nn.Module):
    """
    Define a neural network model.
    """
    def __init__(self, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.optimizer = optim.LBFGS(self.parameters(), lr=1, max_iter=30, tolerance_grad=1e-07,
                                     tolerance_change=1e-09, history_size=100)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, features):
        return self.net(features)

    def predict(self, x, y):
        """
        Predict the function values for given x and y.
        """
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        features = torch.vstack([x, y]).T
        with torch.no_grad():
            label = self.net.forward(features)
            label = label.detach().cpu().numpy().flatten()
        return label


def load_data(filename):
    """Load data from a file."""
    return np.loadtxt(filename)


def create_dataloader(features, labels):
    """Create a DataLoader from features and labels."""
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    return dataloader


def train_model(dataloader, epochs=200):
    """Train the neural network model."""
    net = NeuralNetwork(hidden_size=64).to(device)

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


def evaluate_model(net, n_points, type_name, ax, vmin=None, vmax=None):
    """Evaluate the model and plot the prediction errors."""
    x, y, z = get_uniform_data(n_points)
    z_hat = net.predict(x, y)
    delta = np.abs(z_hat - z) + 1e-6
    return np.min(delta), np.max(delta)


def main():
    datasets = [
        {"name": "train_model", "features": 'dataset/train_features.txt', "labels": 'dataset/train_labels.txt'},
        {"name": "sampled", "features": 'dataset/sampled_features.txt', "labels": 'dataset/sampled_labels.txt'},
        {"name": "managed", "features": 'dataset/managed_features.txt', "labels": 'dataset/managed_labels.txt'}
    ]

    nets = {}
    deltas = []

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    # fig.suptitle('Prediction Error Comparison Across Datasets')

    for dataset in datasets:
        print(f"Processing dataset: {dataset['name']}")
        features = load_data(dataset["features"])
        labels = load_data(dataset["labels"])
        dataloader = create_dataloader(features, labels)
        net = train_model(dataloader)
        nets[dataset['name']] = net

    for i, dataset in enumerate(datasets):
        net = nets[dataset['name']]
        min_delta, max_delta = evaluate_model(net, 1000 * 1000, dataset["name"], axs[i])
        deltas.append((min_delta, max_delta))

    global_min = min(min(d) for d in deltas)
    global_max = max(max(d) for d in deltas)

    # Define boundaries for the discrete colorbar using linear distribution
    n_intervals = 10
    # Create a LogNorm instance with vmin and vmax
    norm = LogNorm(vmin=global_min, vmax=global_max)

    # Use BoundaryNorm for discrete intervals
    cmap = plt.get_cmap('plasma')
    plt.gcf().set_constrained_layout(True)
    plt.tight_layout(h_pad=0.0, w_pad=0.0, rect=[0.10, 0.00, 1.0, 0.85])
    name_list = ['$\mathcal{D}$', '$\mathcal{D}_{M}$', '$\mathcal{D}_{R}$']
    for i, dataset in enumerate(datasets):
        net = nets[dataset['name']]
        x, y, z = get_uniform_data(1000 * 1000)
        z_hat = net.predict(x, y)
        delta = np.abs(z_hat - z) + 1e-8
        sc = axs[i].scatter(x, y, c=delta, cmap=cmap, norm=norm)
        axs[i].axis('equal')
        axs[i].set_title(f"{name_list[i]}\nRMSE:{round(np.mean(delta**2)**0.5, 4)}\n Max Error:{round(np.max(delta), 4)}")
        ticks = np.arange(-3, 3 + 1, 2)  # 这里的5是间距大小，根据您的需求调整
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)

    # Add a colorbar to the figure with scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Adjust power limits for scientific notation
    cb = fig.colorbar(sc, ax=axs.ravel().tolist(), label="Error")

    plt.savefig("comparison.png")
    # plt.show()


if __name__ == '__main__':
    main()