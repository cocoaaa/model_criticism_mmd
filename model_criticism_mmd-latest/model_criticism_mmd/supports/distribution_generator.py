import typing
from typing import Any
from nptyping import NDArray
from sklearn.utils import check_random_state
import numpy as np


def sample_SG(n: int, dim: int, rs=None) -> typing.Tuple[NDArray[(Any, Any), Any], NDArray[(Any, Any), Any]]:
    rs = check_random_state(rs)
    mu = np.zeros(dim)
    sigma = np.eye(dim)
    X = rs.multivariate_normal(mu, sigma, size=n)
    Y = rs.multivariate_normal(mu, sigma, size=n)
    return X, Y


def generate_data(n_train: int, n_test: int):
    np.random.seed(np.random.randint(2 ** 31))
    # X, Y = generate.generate_data(args, n_train + n_test, dtype=floatX)
    # as an example X, Y are from the same distribution.
    X, Y = sample_SG(n_train + n_test, dim=2)
    is_train = np.zeros(n_train + n_test, dtype=bool)
    is_train[np.random.choice(n_train + n_test, n_train, replace=False)] = True
    X_train = X[is_train]
    Y_train = Y[is_train]
    X_test = X[~is_train]
    Y_test = Y[~is_train]

    return X_train, Y_train, X_test, Y_test
