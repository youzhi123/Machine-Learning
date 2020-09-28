import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据，数据描述的是一家连锁店在97个城市的盈利数据，建立城市人口与盈利的线性回归关系
# Population：人口
# Profit：盈利


def data_process():
    data = pd.read_csv('Dataset/ex1data1.txt', names=['Population', 'Profit'])

    # 数据处理，把变量和预测目标取出来，并转换为martix，便于后面计算
    data.insert(0, 'Ones', 1)  # 在第 0 列的位置添加一列
    # print(data.head())
    X = data.loc[:, ['Ones', 'Population']]  # loc根据标签来索引
    y = data.loc[:, ['Profit']]
    # print(X)          # Series类型

    # 转化为mat，便于后面计算
    X = np.mat(X.values)  # 转化为矩阵，类比np.array()
    y = np.mat(y.values)

    return X, y


# 可视化，传入x、y需为array形式
def plot_catter(X, y):
    x1 = np.array(X[:,1])
    y1 = np.array(y)

    plt.scatter(x1, y1, s=10)
    plt.savefig('./figure/1_1.png')


def cost_function(X, y, theta):
    """
    代价函数
    :param X:
    :param y:
    :param theat: 参数 w 和 b
    :return:
    """
    temp = np.power(((X * theta.T) - y), 2)
    cost_f = (1/(2*len(X))) * (np.sum(temp))
    return cost_f


def gradient_descent(X, y, theta, alpha, iters):
    theta_temp = np.mat(np.zeros(2))      # 初始化theta = [o, o]
    parameters = 2                             # 参数个数为2
    cost = np.zeros(iters)               # 初始化每次迭代的损失值 array(0, 0, 0, ...)

    for i in range(iters):
        error = (X * theta.T) - y     # 返回mat()

        for j in range(parameters):
            inner = np.multiply(error, X[:, j])     # 矩阵计算
            # 计算迭代一轮iter的theta_0, theta_1的偏导
            theta_temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(inner))

        theta = theta_temp
        cost[i] = cost_function(X, y, theta)

        return theta, cost


def curve_fitting(X, y, theta):
    x = np.linspace(X[:,1].min(), X[:,1].max(), 100)    # 取100个点
    f = theta[0, 0] + (theta[0,1] * x)

    x1 = np.array(X[:, 1])
    y1 = np.array(y)

    fig, ax = plt.subplots()
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(x1, y1, s=10, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.savefig('./figure/1_2.png')


def loss_epoch_curve(iters, cost_iters):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost_iters, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.savefig("./figure/1_3.png")



# 定义全局变量
alpha = 0.0115
iters = 100
theta = np.mat(np.zeros(2))  # 定义初始参数为 [0, 0]


def run():
    X, y = data_process()
    plot_catter(X, y)
    g, cost_iters = gradient_descent(X, y, theta, alpha, iters)
    print(g)
    cost_f = cost_function(X, y, g)
    print(cost_f)
    loss_epoch_curve(iters, cost_iters)
    curve_fitting(X, y, g)


if __name__ == "__main__":
    run()