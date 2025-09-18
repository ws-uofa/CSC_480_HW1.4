import numpy as np

def sampling(size):
    # Sample uniformly from [0,1]^2
    return np.random.uniform(0, 1, (size, 2))


def labeling(testset):
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
    x = sampling(size)
    y = labeling(x)
    return x, y
