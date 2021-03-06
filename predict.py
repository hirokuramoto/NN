# 訓練済みのニューラルネットワークを使って予測
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 保存したニューラルネットワークを開く
with open('neuralnet.pkl', 'rb') as f:
    dataset = pickle.load(f)
x = dataset.predict(np.array([[1, 1]]))
print(x)

a = np.arange(-2.0, 2.05, 0.05)
b = np.arange(-2.0, 2.05, 0.05)

X, Y = np.meshgrid(a, b)

c = np.array([[dataset.predict(np.array([[ai, bi]]))[0] for ai in a] for bi in b])
Z = c.reshape(81, 81)

fig = plt.figure(figsize=(5, 4.5))
ax = Axes3D(fig, azim = 120)
ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5)

#ax.plot_surface(X, Y, Z, cmap=cm.jet, rstride=1, cstride=1, linewidth=0.3, alpha=0.8)
ax.set_zlim(0, 3600)
plt.show()
