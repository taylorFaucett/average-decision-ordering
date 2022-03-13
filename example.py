import numpy as np

from average_decision_ordering import calc_ado

data_size = 50000  # Size of the test data set
n_pairs = 100000  # Maximum number of pairs
x = list(np.random.rand(data_size))
y = list(np.random.rand(data_size))
targets = list(np.random.randint(2, size=data_size))

# ADO calculated without statistics
print(calc_ado(fx=x, gx=y, target=targets, n_pairs=n_pairs))

# ADO example where you expect perfect similarity (i.e. compare x with x)
print(calc_ado(fx=x, gx=x, target=targets, n_pairs=n_pairs))
