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

        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None


    def forward(self, x):
        self.x = x  # 値を保持
        out = np.dot(x, self.W) + self.b

        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss(object):
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
        dx = (self.y - self.t) / batch_size

        return dx

if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    mask = (x <= 0)
    out = x.copy()
    print(mask)
    print(out)
    out[mask] = 0
    print(out)
