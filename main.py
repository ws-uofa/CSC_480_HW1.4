import numpy as np

from toolset import sampling, labeling
from KNN import trial
import matplotlib.pyplot as plt

bayes_error_rate = 0.2
K = [1,2,4,8,16,32,64]
M = [10,30,100,300,1000,3000]
test_size = 10000
train_size = 3000
trials = 5

def plot(error_rate,k):
    plt.figure(figsize=(8, 5))
    plt.plot(M, error_rate, marker='o', label="kNN Test Error Rate")
    plt.axhline(y=bayes_error_rate, color='red', linestyle='--', label="Bayes Error Rate")
    plt.xscale("log")
    plt.xlabel("Training set size (m)")
    plt.ylabel("Test Error Rate")
    plt.title("learning curve for k-NN (k={})".format(k))
    plt.legend()
    plt.grid(True)
    plt.show()

test_x = sampling(test_size)
test_y = labeling(test_x)

for k in K:
    error_rate = np.zeros((trials,6))
    a = (1/5) * np.ones((5, 1))
    for i in range(trials):
        error_rate[i,:] = trial(k,train_size, test_x, test_y)
    plot(error_rate.T @ a,k)