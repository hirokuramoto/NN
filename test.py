import numpy as np
from load_data import load_data


filename = 'Rosenbrock.csv'
train_size = 100
test_size  = 20
design     = 2
object     = 1
(x_train, t_train), (x_test, t_test) = load_data(filename, train_size, test_size, design, object, normalize=False)

print(x_train[0:5])
print(x_train.mean(axis=0))
print(x_train.std(axis=0))
mean = x_train.mean(axis=0)
std  = x_train.std(axis=0)
x_train = (x_train - mean) / std
print(x_train[0:5])
