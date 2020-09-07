import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from tqdm import tqdm

#from mnist import load_mnist
from load_data import load_data
from two_layer_net import TwoLayerNet
from optimizer import SGD, Momentum, AdaGrad, Adam, RMSprop


def main():
    t1 = time.time()

    #(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # データの読み込み
    filename = 'Rosenbrock.csv'
    train_size = 100
    test_size  = 20
    design     = 2
    object     = 1
    (x_train, t_train), (x_test, t_test) = load_data(filename, train_size, test_size, design, object, normalize=False)

    count = np.array([], dtype = np.int)

    # ハイパーパラメータ
    iters_num = 10000
    batch_size = 20
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list  = []
    test_acc_list   = []
    iter_per_epoch  = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=2, hidden_size=100, output_size=1)
    optimizer = Adam()

    for i in tqdm(range(iters_num)):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        #for key in ('W1', 'b1', 'W2', 'b2'):
        #    network.params[key] -= learning_rate * grad[key]

        optimizer.update(network.params, grad)

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        count = np.append(count, i)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

    # 計算結果の表示
    t2 = time.time()
    elapsed_time = t2 - t1
    print("elapsed_time=", elapsed_time)
    print(train_acc, test_acc)
    print(network.predict(np.array([[1.0, 1.0]])))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'count')
    ax.set_ylabel(r'loss')
    ax.scatter(count, train_loss_list, s=10, c='blue', edgecolors='pink', linewidths=1, marker='o', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
