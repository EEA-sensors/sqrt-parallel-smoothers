import numpy as np

from parsmooth._base import MVNParams


def get_system(dim_x, dim_y):
    m = np.random.randn(dim_x)
    cholP = np.random.rand(dim_x, dim_x)
    cholP[np.triu_indices(dim_x, 1)] = 0.
    P = cholP @ cholP.T

    cholR = np.random.rand(dim_y, dim_y)
    cholR[np.triu_indices(dim_y, 1)] = 0.
    R = cholR @ cholR.T

    H = np.random.randn(dim_y, dim_x)
    c = np.random.randn(dim_y)
    y = np.random.randn(dim_y)

    chol_x = MVNParams(m, None, cholP)
    x = MVNParams(m, P)
    return x, chol_x, H, R, cholR, c, y