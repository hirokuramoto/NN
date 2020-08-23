import pandas as pd
import numpy as np

data = pd.read_csv('Rosenbrock.csv', header = 0) # header = 0でヘッダー行を読み飛ばし
x_train = np.array(data.iloc[0:100,   0:2])
t_train = np.array(data.iloc[0:100,   2])
x_test  = np.array(data.iloc[100:120, 0:2])
t_test  = np.array(data.iloc[100:120, 2])

print(x_train.shape[0])
print(t_train.shape[0])
print(x_test)
print(t_test)
print(t_test.ndim)
