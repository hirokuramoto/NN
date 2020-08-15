from collections import OrderedDict

import numpy as np

from functions import *
from gradient import numerical_gradient

class TwoLayerNet(object):
    '''2層のニューラルネット
    input
        input_size  : 入力層のニューロンの数
        hidden_size : 隠れ層のニューロンの数
        output_size : 出力層のニューロンの数
    '''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']   = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t):
        '''損失関数の計算
        input
            x : 入力データ
            t : 教師データ
        '''
        y = self.predict(x)

        return cross_entropy_error(y, t)


    def accuracy(self, x, t):
        '''認識精度の計算
        input
            x : 入力データ
            t : 教師データ
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1) # axis=1で行ごとの最大値の列番号インデックスを返す
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        '''認識精度の計算
        input
            x : 入力データ
            t : 教師データ
        '''
        loss_W = lambda W: self.loss(x, t)

        grads={}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


if __name__ == '__main__':
    test = TwoLayerNet(784, 100, 10)
    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    print(test.accuracy(x, t))
