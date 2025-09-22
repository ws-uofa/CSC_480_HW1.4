import numpy as np

def sampling(size):
    """
       Uniformly sample points from the 2D unit square [0,1]^2.

       Args:
           size (int): Number of samples to generate.

       Returns:
           np.ndarray: Array of shape (size, 2) containing sampled points.
    """
    return np.random.uniform(0, 1, (size, 2))


def labeling(testset):
    """
       Assign labels (binary classification) to a set of 2D points
       based on a probability matrix.

       Args:
           testset (np.ndarray): Array of shape (n_samples, 2).

       Returns:
           np.ndarray: Boolean array of labels (True/False).
    """
    # Initialize y as boolean array of labels
    y = np.zeros(testset.shape[0], dtype=bool)

    # Probability Matrix
    A = np.array([[.1, .2, .2],
                  [.2, .4, .8],
                  [.2, .8, .9]])

    for i in range(testset.shape[0]):
        # Mapping x1 and x2 into {0,1,2}
        if testset[i, 0] < 1 / 3:
            x1 = 0
        elif testset[i, 0] < 2 / 3:
            x1 = 1
        else:
            x1 = 2

        if testset[i, 1] < 1 / 3:
            x2 = 0
        elif testset[i, 1] < 2 / 3:
            x2 = 1
        else:
            x2 = 2

        # Assign label based on Bernoulli trial
        y[i] = np.random.binomial(1, A[x1, x2])
    return y


def create_dataset(size):
    """
       Create a dataset of (X, y) pairs.

       Args:
           size (int): Number of samples.

       Returns:
           tuple: (X, y)
               - X (np.ndarray): Features of shape (size, 2).
               - y (np.ndarray): Labels of shape (size,).
    """
    x = sampling(size)
    y = labeling(x)
    return x, y
