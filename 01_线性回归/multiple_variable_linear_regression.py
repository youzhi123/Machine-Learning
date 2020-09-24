import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
data = pd.read_csv('Dataset/ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])

# 定义全局变量
theta = np.mat(np.zeros(data.shape[1]))
iters = 3000
alpha = 0.005


def data_process(data):
    # 标准化
    data = (data - data.mean()) / data.std()

    data.insert(0, 'Ones', 1)

    X1 = data.iloc[:, 0:3]
    y1 = data.iloc[:, [-1]]

    X = np.mat(X1.values)
    y = np.mat(y1.values)

    return X, y

def data_plot(X, y):
    Size = np.array(X[:, 1])
    Bedrooms = np.array(X[:, 2])
    Price = np.array(y[:, 0])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Size, Bedrooms, Price, color='indianred')
    plt.savefig('./figure/2_1.png')


def cost_function(X, y, theta):
    temp = np.power((X * theta.T - y), 2)
    cost_f = (1/(2 * len(X))) * (np.sum(temp))
    return cost_f


def gradient_descent(X, y, theta, alpha, iters):
    theta_temp = np.mat(np.zeros(3))
    parameters = 3
    cost = np.zeros(iters)

    for i in range(iters):
        error = X * theta.T - y
        for j in range(parameters):
            inner = np.multiply(error, X[:, j])
            theta_temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(inner))

        theta = theta_temp
        cost[i] = cost_function(X, y, theta)

    return theta, cost


def curve_fit(X, y, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Size = np.array(X[:, 1])
    Bedrooms = np.array(X[:, 2])
    Price = np.array(y[:, 0])

    x1 = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    x2 = np.linspace(X[:,2].min(), X[:,2].max(), 100)
    M, N = np.meshgrid(x1, x2)
    # f = theta[0, 0] + (theta[0, 1] * x1) + (theta[0, 1] * x2)
    Z = np.array([theta[0, 0] + theta[0, 1] * d + theta[0, 2] * p for d, p in zip(np.ravel(M), np.ravel(N))]).reshape(M.shape)
    ax.plot_surface(M, N, Z)
    ax.scatter(Size, Bedrooms, Price)

    ax.set_xlabel('Size')
    ax.set_ylabel('Bedrooms')
    ax.set_zlabel('Price')

    plt.savefig('./figure/2_2.png')


def loss_epoch_curve(cost_iter, iters):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost_iter, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.savefig('./figure/2_3.png')

def run():
    X, y = data_process(data)
    data_plot(X, y)
    g, cost_iter = gradient_descent(X, y, theta, alpha, iters)
    print(g)
    curve_fit(X, y, g)
    loss_epoch_curve(cost_iter, iters)
    cost_f = cost_function(X, y, g)
    print(cost_f)

if __name__ == "__main__":
    run()