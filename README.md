# CSC_480_HW1.4

# k-NN Learning Curve Simulation

This project simulates the performance of **k-Nearest Neighbors (k-NN)** classifiers on a synthetic 2D dataset.  
It visualizes the learning curve (test error rate vs. training set size) for different values of `k`, and compares them against the Bayes error rate.

---

## Project Structure

- **Toolset.py**  
  Contains helper functions:
  - `sampling(size)` → Sample uniformly from `[0,1]^2`
  - `labeling(testset)` → Assign probabilistic labels using a predefined probability matrix
  - `create_dataset(size)` → Generate `(X, y)` dataset pairs

- **KNN.py**  
  Implements training and evaluation of k-NN:
  - `compute_error()` → Compute error rate
  - `trial()` → Run one experiment for a given `k` and training size
  - `plot()` → Plot the learning curve

- **main.py**  
  Runs multiple trials for various values of `k` and plots averaged learning curves.

---

## Requirements

Install dependencies with:

```bash
pip install numpy scikit-learn matplotlib
