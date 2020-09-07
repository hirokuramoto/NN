import numpy as np
from load_data import load_data


filename = 'Rosenbrock.csv'
train_size = 100
test_size  = 20
design     = 2
object     = 1
(x_train, t_train), (x_test, t_test) = load_data(filename, train_size, test_size, design, object, normalize=True)

print(x_train)
print(t_train[np.arange(20), t_train])
