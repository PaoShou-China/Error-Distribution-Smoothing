import pandas as pd
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(file_path):
    """Load results file"""
    try:
        return pd.read_csv(file_path, index_col=0)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def save_data_to_txt(all_results, metric, dataset_order, models, save_path):
    """Save plot data to txt file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f"# {metric} Comparison Across Datasets\n\n")
        header = "Model"
        for dataset in dataset_order:
            header += f"\t{dataset}"
        f.write(header + "\n")
        for model in models:
            line = f"{model}"
            for dataset in dataset_order:
                if model in all_results[dataset].index:
                    value = all_results[dataset].loc[model, metric]
                    line += f"\t{value:.6f}"
                else:
                    line += "\tN/A"
            f.write(line + "\n")

def plot_comparison(all_results, metric, title, ylabel):
    """Plot comparison of model metrics across different datasets (style consistent with dim0)"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 24,
        'figure.titlesize': 18
    })
    model_abbr = {
        'Ridge': 'Ridge',
        'BayesianRidge': 'BR',
        'DecisionTreeRegressor': 'DT',
        'RandomForestRegressor': 'RF',
        'Lasso': 'Lasso',
        'SVR': 'SVR',
        'XGBRegressor': 'XGB'
    }
    fig, ax = plt.subplots(figsize=(15, 7))
    dataset_order = ['Original', 'SMOGN', 'Sampled', 'EDS']
    dataset_name_display = ['$\mathcal{D}$', '$\mathcal{D}_{S}$', '$\mathcal{D}_{M}$', '$\mathcal{D}_{R}$']
    models = list(all_results[list(all_results.keys())[0]].index)
    abbr_models = [model_abbr.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.2
    colors = ['#F39DA0', '#95BCE5', '#E84445', '#1999B2']
    for i, dataset_name in enumerate(dataset_order):
        dataset_data = []
        for model in models:
            if model in all_results[dataset_name].index:
                dataset_data.append(all_results[dataset_name].loc[model, metric])
            else:
                dataset_data.append(np.nan)
        ax.bar(x + i*width, dataset_data, width, label=dataset_name_display[i], color=colors[i], alpha=0.8)
    ax.set_xlabel('Models', labelpad=10, fontsize=28)
    ax.set_ylabel(ylabel, labelpad=10, fontsize=28)
    ax.set_xticks(x + (len(dataset_order)-1)/2*width)
    ax.set_xticklabels(abbr_models)
    plt.subplots_adjust(bottom=0.2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_yscale('log')
    all_values = []
    for dataset_name in dataset_order:
        for model in models:
            if model in all_results[dataset_name].index:
                value = all_results[dataset_name].loc[model, metric]
                if not np.isnan(value) and value > 0:
                    all_values.append(value)
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        ax.set_ylim([min_val * 0.5, max_val * 1.5])
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    # save_data_path = f'4/cartpole/visualization/{metric.lower().replace(" ", "_")}_comparison_dim1.txt'
    # save_data_to_txt(all_results, metric, dataset_order, models, save_data_path)
    save_path = f'results/figures/3/{metric.lower().replace(" ", "_")}_comparison_dim1.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_comparison(all_results):
    """Plot model training time comparison (style consistent with dim0)"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 24,
        'figure.titlesize': 18
    })
    model_abbr = {
        'Ridge': 'Ridge',
        'BayesianRidge': 'BR',
        'DecisionTreeRegressor': 'DT',
        'RandomForestRegressor': 'RF',
        'Lasso': 'Lasso',
        'SVR': 'SVR',
        'XGBRegressor': 'XGB'
    }
    fig, ax = plt.subplots(figsize=(15, 7))
    dataset_order = ['Original', 'SMOGN', 'Sampled', 'EDS']
    dataset_name_display = ['$\mathcal{D}$', '$\mathcal{D}_{S}$', '$\mathcal{D}_{M}$', '$\mathcal{D}_{R}$']
    models = list(all_results[list(all_results.keys())[0]].index)
    abbr_models = [model_abbr.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.2
    colors = ['#F39DA0', '#95BCE5', '#E84445', '#1999B2']
    for i, dataset_name in enumerate(dataset_order):
        dataset_data = []
        for model in models:
            if model in all_results[dataset_name].index:
                dataset_data.append(all_results[dataset_name].loc[model, 'Time Taken'])
            else:
                dataset_data.append(np.nan)
        ax.bar(x + i*width, dataset_data, width, label=dataset_name_display[i], color=colors[i], alpha=0.8)
    ax.set_xlabel('Models', labelpad=10, fontsize=28)
    ax.set_ylabel('Training Time (s)', labelpad=10, fontsize=28)
    ax.set_xticks(x + (len(dataset_order)-1)/2*width)
    ax.set_xticklabels(abbr_models)
    plt.subplots_adjust(bottom=0.2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_yscale('log')
    all_values = []
    for dataset_name in dataset_order:
        for model in models:
            if model in all_results[dataset_name].index:
                value = all_results[dataset_name].loc[model, 'Time Taken']
                if not np.isnan(value) and value > 0:
                    all_values.append(value)
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        ax.set_ylim([min_val * 0.5, max_val * 1.5])
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    # save_data_path = '4/cartpole/visualization/training_time_comparison_dim1.txt'
    # save_data_to_txt(all_results, 'Time Taken', dataset_order, models, save_data_path)
    save_path = 'results/figures/3/training_time_comparison_dim1.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set Chinese font for labels (if needed)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # For proper display of Chinese labels
    plt.rcParams['axes.unicode_minus'] = False  # For proper display of minus sign
    sns.set_style("whitegrid")
    # Define list of models to display (consistent with dim0)
    selected_models = [
        'Ridge',
        'BayesianRidge',
        'DecisionTreeRegressor',
        'RandomForestRegressor',
        'Lasso',
        'SVR',
        'XGBRegressor'
    ]
    # Define result file paths
    results_files = {
        'Original': 'results/lazypredict/cartpole/train_dim1.csv',
        'SMOGN': 'results/lazypredict/cartpole/smogn_dim1.csv',
        'Sampled': 'results/lazypredict/cartpole/sampled_dim1.csv',
        'EDS': 'results/lazypredict/cartpole/eds_dim1.csv'
    }
    # Load all results, keep only selected models
    all_results = {}
    for dataset_name, file_path in results_files.items():
        results = load_results(file_path)
        if results is not None:
            all_results[dataset_name] = results.loc[selected_models]
    # Create comparison table
    comparison_data = []
    for model_name in selected_models:
        row = [model_name]
        for dataset_name in results_files.keys():
            results = all_results[dataset_name]
            if model_name in results.index:
                rmse = results.loc[model_name, 'RMSE']
                max_err = results.loc[model_name, 'Max Error']
                time_taken = results.loc[model_name, 'Time Taken']
                row.extend([f"{rmse:.4f}", f"{max_err:.4f}", f"{time_taken:.2f}s"])
            else:
                row.extend(['N/A', 'N/A', 'N/A'])
        comparison_data.append(row)
    headers = ['Model']
    for dataset_name in results_files.keys():
        headers.extend([f'{dataset_name} RMSE', f'{dataset_name} Max Error', f'{dataset_name} Time'])
    print("\nComparison of Selected Models Across Datasets:")
    print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
    print("\nBest Models for Each Dataset:")
    for dataset_name, results in all_results.items():
        best_rmse_model = results.loc[results['RMSE'].idxmin()]
        best_max_err_model = results.loc[results['Max Error'].idxmin()]
        print(f"\n{dataset_name} Dataset:")
        print(f"Best RMSE: {best_rmse_model.name} (RMSE: {best_rmse_model['RMSE']:.4f}, Max Error: {best_rmse_model['Max Error']:.4f}, Time: {best_rmse_model['Time Taken']:.2f}s)")
        print(f"Best Max Error: {best_max_err_model.name} (RMSE: {best_max_err_model['RMSE']:.4f}, Max Error: {best_max_err_model['Max Error']:.4f}, Time: {best_max_err_model['Time Taken']:.2f}s)")
    print("\nGenerating visualization plots...")
    plot_comparison(all_results, 'RMSE', 'RMSE Comparison Across Datasets (Selected Models)', 'RMSE')
    plot_comparison(all_results, 'Max Error', 'Max Error Comparison Across Datasets (Selected Models)', 'Max Error')
    plot_time_comparison(all_results)

if __name__ == "__main__":
    main() 