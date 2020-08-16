from collections import OrderedDict

import numpy as np

from layers import *
from gradient import numerical_gradient

class TwoLayerNet(object):
    '''2層のニューラルネット
    input
        input_size  : 入力層のニューロンの数
        hidden_size : 隠れ層のニューロンの数
        output_size : 出力層のニューロンの数
    '''

    def __init__(self, input_size, hidden_size, output_size):
        # 重みの初期化(活性化関数がReluなので、Heの初期値を利用)
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']   = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        '''認識（推論）の実行
        input:
            x(np.array) : 画像データ
        output:
            x(np.array) : スコア（出力層への入力値）
        '''
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t):
        '''損失関数の計算
        input:
            x(np.array) : スコア
            t(np.array) : 教師データ
        output:
            np.array : Softmax-with-Lossレイヤからの出力（確率値）
        '''
        y = self.predict(x)

        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        '''認識精度の計算
        input:
            x(np.array) : 画像データ
            t(np.array) : 教師データ
        output:
            accuracy(float) : 認識精度
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1) # axis=1で行ごとに最大値の列番号インデックスを返す

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0]) # 最大値のインデックス番号同士を比較
        return accuracy


    def numerical_gradient(self, x, t):
        '''重みパラメータに対する勾配を数値微分によって求める
        input:
            x(np.array) : 画像データ
            t(np.array) : 教師データ
        output:
            grads(dict) : 重みパラメータ
        '''
        loss_W = lambda W: self.loss(x, t)

        grads={}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


    def gradient(self, x, t):
        ''' 重みパラメータに対する勾配を誤差逆伝播法によって求める
        input :
            x(np.array) :
            t(np.array) :
        output :
            grads(dict) : 重みパラメータ
        '''
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values()) # layers(dict型)の値の一覧をlist化
        layers.reverse()    # reverse()メソッド : listを逆順に並べ替え
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


if __name__ == '__main__':
    test = TwoLayerNet(784, 100, 10)
    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    print(test.accuracy(x, t))
