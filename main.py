import numpy as np

from Toolset import sampling, labeling
from KNN import trial,plot

bayes_error_rate = 0.2
K = [1,2,4,8,16,32,64]
test_size = 10000
train_size = 3000
trials = 5

# Generate a fixed test set
test_x = sampling(test_size)
test_y = labeling(test_x)

# Run experiments for each k
for k in K:
    error_rate = np.zeros((trials,6))
    a = (1/5) * np.ones((5, 1)) # Weight matrix
    for i in range(trials):
        error_rate[i,:] = trial(k,train_size, test_x, test_y)
    plot(error_rate.T @ a,k)