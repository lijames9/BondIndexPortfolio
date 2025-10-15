import numpy as np


def pca(rets, p=None):
    """
    Takes in the returns T x n and outputs the mu and Q of dataset

    :param rets: returns in T, n shape
    :param p: number of principal components to use
    :return: mu and Q for dataset
    """

    # set up and get dimensions
    R = np.array(rets)
    T = R.shape[0]
    n = R.shape[1]

    # Get the mean of each column
    r_bar = 1 / T * np.matmul(R.T, np.ones((T, 1)))

    # Subtract the mean of each column from that column, getting mean 0 for all columns
    R_bar = R - np.matmul(np.ones((T, 1)), r_bar.T)

    # Calculate the biased covariance matrix
    Q = 1 / T * np.matmul(R_bar.T, R_bar)

    # Perform Spectral decomposition of Q
    Lambda, Gamma = np.linalg.eigh(Q)

    # Generate principal component matrix
    P = np.matmul(R_bar, Gamma)

    # TODO change to proportion of variance
    if p is None:
        p = n

    # Select the first p elements
    P_1 = P[:, 0:p]
    Gamma_1 = Gamma[:, 0:p]

    # Create data matrix
    X = np.concatenate((np.ones((T, 1)), P_1), axis=1)

    # TODO change this to regress X against R_bar
    V = Gamma_1
    alpha = r_bar
    B = np.concatenate((alpha.T, V.T), axis=0)

    # Get errors
    ep = R - np.matmul(X, B)
    norms = np.linalg.norm(ep, axis=0)
    D = np.diag(norms)

    # Get factors
    f_bar = np.mean(P_1, axis=0)
    f_bar = f_bar.reshape((p, 1))
    F = np.cov(P_1.T)

    # Parameter estimation
    mu = alpha + np.matmul(V, f_bar)
    Q = np.matmul(np.matmul(V, F), V.T) + D

    return mu, Q
