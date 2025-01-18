import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, max_error
import time
import matplotlib.pyplot as plt

# 加载数据的函数保持不变
def load_data(features_path, labels_path):
    features = np.loadtxt('dataset/' + features_path)
    labels = np.loadtxt('dataset/' + labels_path)

    # Check if features is a vector and convert it to a column vector
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Check if labels is a vector and convert it to a column vector
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    return features, labels

# 保存评估结果到txt文件中的函数
def save_evaluation_results(file_path, epoch, rmse, max_err):
    with open(file_path, 'a') as f:
        f.write(f"{epoch},{rmse:.5f},{max_err:.5f}\n")

# 训练和评估MLP模型的函数
def evaluate_mlp(train_features, train_labels, test_features, test_labels, max_epochs=3000, eval_interval=10, results_file='evaluation_results.txt'):
    # 如果文件存在则清空文件内容
    open(results_file, 'w').close()

    # 标准化特征数据（仅在训练集上fit，在测试集上transform）
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)

    # 创建 MLP 模型并初始化
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', solver='adam',
                       alpha=0.0001, batch_size=128, learning_rate='constant',
                       learning_rate_init=0.0005, power_t=0.5, max_iter=eval_interval,
                       shuffle=True, random_state=3704, tol=1e-8,
                       verbose=False, warm_start=True, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False,
                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-8, n_iter_no_change=10, max_fun=15000)

    for epoch in range(0, max_epochs, eval_interval):
        # 开始计时
        start_time = time.time()

        # 训练模型
        mlp.fit(scaled_train_features, train_labels.ravel())

        # 结束计时
        end_time = time.time()

        # 预测测试集并计算 RMSE 和 最大误差
        y_pred = mlp.predict(scaled_test_features)
        rmse = mean_squared_error(test_labels, y_pred, squared=False)
        max_err = max_error(test_labels, y_pred)

        # 保存评估结果到文件
        save_evaluation_results(results_file, epoch + eval_interval, rmse, max_err)

    print(f"Training completed.")

# 读取评估结果并绘图的函数
def plot_evaluation_results(results_file, dataset_name):
    epochs, rmses, max_errors = [], [], []

    with open(results_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            epoch, rmse, max_err = map(float, line.strip().split(','))
            epochs.append(epoch)
            rmses.append(rmse)
            max_errors.append(max_err)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, rmses, label='RMSE')
    plt.plot(epochs, max_errors, label='Max Error')
    plt.title(f'{dataset_name} Dataset: Metrics vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

# 使用示例：调用函数并传入训练集和测试集的路径
if __name__ == "__main__":
    for head in ['train', 'sampled', 'managed']:
        print(f"Evaluating {head} dataset over iterations...")

        # 加载所有数据
        train_features, train_labels = load_data(head + '_features.txt', head + '_labels.txt')
        test_features, test_labels = load_data('test_features.txt', 'test_labels.txt')

        # 定义结果文件名
        results_file = f'{head}_evaluation_results.txt'

        # 评估模型并获取每轮迭代的结果
        evaluate_mlp(train_features, train_labels, test_features, test_labels, results_file=results_file)

        # 绘制 RMSE 和 最大误差随迭代次数变化的图
        # plot_evaluation_results(results_file, head)

        print(f"\nFinal Results for {head} dataset have been saved to {results_file}.\n")