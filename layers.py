import numpy as np

from functions import *

class Relu(object):
    '''Reluレイヤ
        順伝播時の入力が
            > 0 ：逆伝播時は上流の値をそのまま流す
            0 >=：逆伝播時は信号を流さない
    '''
    def __init__(self):
        self.mask = None


    def forward(self, x):
        self.mask = (x <= 0) # 順伝播時に０以下ならTrue
        out = x.copy()
        out[self.mask] = 0   # Trueの要素（負の要素）を０に変換

        return out


    def backward(self,dout):
        dout[self.mask] = 0  # Trueの要素（負の要素）を０に変換
        dx = dout

        return dx


class Sigmoid(object):
    '''Sigmoidレイヤ
        順伝播時：入力xに対して y = 1/(1+exp(-x))
        逆伝播時：入力doutに対して dx = dout * y(1-y)
    '''
    def __init__(self):
        self.out = None # 順伝播時のデータを逆伝播時に使用するため値を保持する


    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out


    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine(object):
    '''Affine(行列積)レイヤ
    '''
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss(object):
    '''Softmax関数と交差エントロピー誤差のレイヤ（分類問題の出力層）
    '''
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmaxの出力
        self.t = None # 教師データ(one-hot vector)


    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss


    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class IdentityWithLoss(object):
    '''恒等関数と２乗和誤差のレイヤ（回帰問題の出力層）
    '''
    def __init__(self):
        self.loss = None # 損失
        self.y = None # 恒等関数の出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = identity_function(x)
        self.loss = sum_squared_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class Dropout(object):
    '''http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization(object):
    '''http://arxiv.org/abs/1502.03167
    '''
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('Rosenbrock.csv', header = 0) # header = 0でヘッダー行を読み飛ばし
    x_train = np.array(data.iloc[0:100,   0:2])
    t_train = np.array(data.iloc[0:20,   2])
    x_test  = np.array(data.iloc[100:120, 0:2])
    t_test  = np.array(data.iloc[100:120, 2])

    test = IdentityWithLoss()
    test.forward(t_train, t_test)
    print(test.backward())
