import pandas as pd
import numpy as np

data = pd.read_csv('Rosenbrock.csv', header = 0) # header = 0でヘッダー行を読み飛ばし
x_train = np.array(data.iloc[0:100,   0:2])
t_train = np.array(data.iloc[0:100,   2])
x_test  = np.array(data.iloc[100:120, 0:2])
t_test  = np.array(data.iloc[100:120, 2])

train_size = 100
batch_size = 20

batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
print(x_batch.shape)
print(t_batch.shape)
print(np.arange(20))

dx = x_train.copy()
dx[np.arange(batch_size), t_train] -= 1
print(dx)
