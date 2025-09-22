from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Toolset import create_dataset

# Training set sizes to explore
M = [10,30,100,300,1000,3000]
# Theoretical Bayes error rate (pre-calculated)
bayes_error_rate = 0.2

def plot(error_rate,k):
    """
        Plot learning curve for k-NN.

        Args:
            error_rate (list): List of test error rates corresponding to M.
            k (int): Number of neighbors used in k-NN.
    """
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
    """
       Compute classification error rate.

       Args:
           knn (KNeighborsClassifier): Trained classifier.
           testset_x (np.ndarray): Test features.
           testset_y (np.ndarray): Test labels.

       Returns:
           float: Error rate.
    """
    p = knn.predict(testset_x)
    return 1-accuracy_score(testset_y, p)

def trial(k,train_size,testset_x,testset_y):
    """
       Run a single trial of training and evaluation for k-NN.

       Args:
           k (int): Number of neighbors.
           train_size (int): Maximum training set size to sample.
           testset_x (np.ndarray): Test features.
           testset_y (np.ndarray): Test labels.

       Returns:
           list: Error rates for different training sizes M.
    """
    error_rate = []
    S_x,S_y = create_dataset(train_size)
    knn = KNeighborsClassifier(n_neighbors=k)

    for m in M:
        km = min(k, m) # Ensure k <= m
        knn.set_params(n_neighbors=km)
        knn.fit(S_x[:m,:],S_y[:m])
        error_rate.append(compute_error(knn, testset_x, testset_y))

    return error_rate
