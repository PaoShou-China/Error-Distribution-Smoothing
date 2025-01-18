import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Used for simple moving average


def load_evaluation_results(file_path):
    """
    Load evaluation results from a specified file path.
    """
    epochs, rmses, max_errors = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            epoch, rmse, max_err = map(float, line.strip().split(','))
            epochs.append(epoch)
            rmses.append(rmse)
            max_errors.append(max_err)
    return epochs, rmses, max_errors


def smooth_data(data, method='moving_average', window_size=10, alpha=0.6):
    """
    Smooth data using moving average or exponential weighted moving average.
    """
    if method == 'moving_average':
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    elif method == 'ewma':  # Exponentially Weighted Moving Average
        return pd.Series(data).ewm(alpha=alpha, adjust=False).mean()


# Function to plot evaluation results
def plot_evaluation_results(files, labels, title_prefix, smoothing_method='moving_average', window_size=40, alpha=0.6):
    """
    Plot evaluation results from multiple files.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Set color list
    colors = ['blue', 'green', 'red']

    # Set global style
    plt.style.use('seaborn-whitegrid')

    # Set larger font sizes
    font_size = {'title': 16, 'label': 16, 'legend': 16, 'ticks': 16}

    for ax in axes:
        ax.grid(True, which="both", ls="--", linewidth=1.5)  # Add grid lines

        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=font_size['ticks'])
        ax.tick_params(axis='both', which='minor', labelsize=font_size['ticks'] - 2)  # 稍微小一点的次级刻度标签

    for file, label, color in zip(files, labels, colors):
        epochs, rmses, max_errors = load_evaluation_results(file)

        # Smooth data
        y_smooth_rmse = smooth_data(rmses, method=smoothing_method, window_size=window_size, alpha=alpha)
        y_smooth_max_error = smooth_data(max_errors, method=smoothing_method, window_size=window_size, alpha=alpha)

        # Plot smoothed lines with shaded areas for confidence intervals or error ranges
        axes[0].fill_between(epochs, y_smooth_rmse - 0.2 * y_smooth_rmse, y_smooth_rmse + 0.2 * y_smooth_rmse,
                             color=color, alpha=0.2)
        axes[0].semilogy(epochs, y_smooth_rmse, label=label, color=color, linewidth=3)  # 增加线宽

        axes[1].fill_between(epochs, y_smooth_max_error - 0.2 * y_smooth_max_error,
                             y_smooth_max_error + 0.2 * y_smooth_max_error, color=color, alpha=0.2)
        axes[1].semilogy(epochs, y_smooth_max_error, label=label, color=color, linewidth=3)  # 增加线宽

        for ax in axes:
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins='auto'))
            ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=5))  # 对数坐标轴上的刻度数量
            ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(0.0, 0.1, 0.02)))
            ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x) if x >= 0.1 else ''))

    # Set titles and labels
    for idx, metric in enumerate(['RMSE', 'Max Error']):
        axes[idx].set_xlabel('Epoch', fontsize=font_size['label'])
        axes[idx].set_ylabel(f'{metric}', fontsize=font_size['label'])
        axes[idx].legend(fontsize=font_size['legend'])

    # Adjust layout to prevent overlap and increase spacing between plots
    plt.subplots_adjust(wspace=0.3)  # wspace parameter controls the width spacing between subplots
    plt.tight_layout()  # Automatically adjust subplot parameters to fill the entire image area

    plt.show()


if __name__ == "__main__":
    files = ['train_evaluation_results.txt', 'sampled_evaluation_results.txt', 'managed_evaluation_results.txt']
    labels = ['$\mathcal{D}$', '$\mathcal{D}_M$', '$\mathcal{D}_R$']

    # Call the plotting function with file list, labels, and smoothing parameters
    plot_evaluation_results(files, labels, "Comparison of Datasets", smoothing_method='moving_average', window_size=40,
                            alpha=0.0)

