from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from toolset import create_dataset

M = [10,30,100,300,1000,3000]
bayes_error_rate = 0.2

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

def compute_error(knn, testset_x, testset_y):
    p = knn.predict(testset_x)
    return 1-accuracy_score(testset_y, p)

def trial(k,train_size,testset_x,testset_y):
    error_rate = []
    S_x,S_y = create_dataset(train_size)
    knn = KNeighborsClassifier(n_neighbors=k)
    for m in M:
        km = min(k, m)
        knn.set_params(n_neighbors=km)
        knn.fit(S_x[:m,:],S_y[:m])
        error_rate.append(compute_error(knn, testset_x, testset_y))
    return error_rate
