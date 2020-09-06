import numpy as np
import pandas as pd

y = np.array([1.0, 2.0])
t = np.array([2.0, 3.0])

if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

# データがバッチの場合
batch_size = y.shape[0]
x = 0.5 * np.sum((y - t)**2) / batch_size

print(y)
print(t)
print(batch_size)

print(x)
