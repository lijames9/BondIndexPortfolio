import numpy as np
import scipy


def black_litterman(Sigma, x_mkt, r_f, rets, investor_views, tau):
    """
    returns adjusted expected returns from market weights and investor views

    :param Sigma: covariance matrix
    :param x_mkt: market weights
    :param r_f: risk_free rate
    :param rets: table of returns of assets
    :param investor_views: matrix of investor views
    :param tau: certainty level, in [0.01, 0.05]
    :return: adjusted expected returns
    """
    volatility_mkt = np.matmul(x_mkt.T, np.matmul(Sigma, x_mkt))

    mu_raw = scipy.stats.gmean(1 + rets, axis=0) - 1

    lambda_ = (np.matmul(mu_raw.T, x_mkt) - r_f) / volatility_mkt

    # Get pi
    pi = lambda_ * np.matmul(Sigma, x_mkt)

    P, q = investor_views

    k = len(q)
    Omega = np.zeros((k, k))

    for i in range(k):
        P_i = P[i, :]
        Omega[k, k] = tau * np.matmul(P_i, np.matmul(Sigma, P_i.T))

    # Calculate mu_bar
    tQ_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)
    POP = np.matmul(P.T, np.matmul(Omega_inv, P))
    POq = np.matmul(P.T, np.matmul(Omega_inv, q))

    first = tQ_inv + POP
    second = np.matmul(tQ_inv, pi) + POq

    mu_bar = np.matmul(np.linalg.inv(first), second)

    return mu_bar
