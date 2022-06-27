import math
from functools import partial
from typing import Tuple, Union, List

import jax
import jax.numpy as jnp
import numpy as np

from parsmooth._base import MVNStandard, FunctionalModel, ConditionalMomentsModel, MVNSqrt
from parsmooth.linearization._sigma_points import SigmaPoints, linearize_functional, linearize_conditional


def linearize(model: Union[FunctionalModel, ConditionalMomentsModel],
              x: Union[MVNSqrt, MVNStandard], order: int = 3):
    """
    Gauss-Hermite linearization for a non-linear function f(x, q). While this may look inefficient for functions with
    additive noise, JAX relies on XLA which compresses linear operations. This means that in practice our code will only
    slow down tracing (compilation) time and not run time.

    Parameters
    ----------
    model: Union[FunctionalModel, ConditionalMomentsModel]
        The function to be called on x and q
    x: Union[MVNSqrt, MVNStandard]
        x-coordinate state at which to linearize f
    order: int, optional
        Order of the Gauss-Hermite integration method. Default is 3.

    Returns
    -------
    F_x, F_q, res: jnp.ndarray
        The linearization parameters
    chol_q or cov_q: jnp.ndarray
        Either the cholesky or the full-rank modified covariance matrix.
    """
    get_sigma_points = lambda mvn: _get_sigma_points(mvn, order)
    if isinstance(model, FunctionalModel):
        f, q = model
        return linearize_functional(f, x, q, get_sigma_points)
    conditional_mean, conditional_covariance_or_cholesky = model
    return linearize_conditional(conditional_mean, conditional_covariance_or_cholesky, x, get_sigma_points)


@partial(jax.jit, static_argnums=(1,))
def _get_sigma_points(
        mvn: MVNSqrt, order: int
) -> Tuple[SigmaPoints, jnp.ndarray]:
    """ Computes the sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim

    Parameters
    ----------
    mvn: MVNSqrt
        Mean and Sqrt covariance of the distribution
    order: int
        Order of the Gauss-Hermite integration method

    Returns
    -------
    out: SigmaPoints
        sigma points for the spherical cubature transform
    xi: jnp.ndarray
        Unit sigma points vectors used
    """
    mean, chol = mvn
    n_dim = mean.shape[0]
    wm, wc, xi = _gauss_hermite_weights(n_dim, order)
    sigma_points = mean[None, :] + (mvn.chol @ xi).T

    return SigmaPoints(sigma_points, wm, wc)


def _gauss_hermite_weights(n_dim: int, order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the weights associated with the Gauss--Hermite quadrature method.
    The Hermite polynomial is in the physician version
    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem
    order: int, optional, default is 3
        The order of Hermite polynomial
    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    xi: np.ndarray
        Orthogonal vectors
    References
    ----------
    .. [1] Simo Särkkä.
       *Bayesian Filtering and Smoothing.*
       In: Cambridge University Press 2013.
    """
    n = n_dim
    p = order

    hermite_coeff = _hermite_coeff(p)
    hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

    table = np.zeros(shape=(n, p ** n))

    w_1d = np.zeros(shape=(p,))
    for i in range(p):
        w_1d[i] = (2 ** (p - 1) * np.math.factorial(p) * np.sqrt(np.pi) /
                   (p ** 2 * (np.polyval(hermite_coeff[p - 1],
                                         hermite_roots[i])) ** 2))

    # Get roll table
    for i in range(n):
        base = np.ones(shape=(1, p ** (n - i - 1)))
        for j in range(1, p):
            base = np.concatenate([base,
                                   (j + 1) * np.ones(shape=(1, p ** (n - i - 1)))],
                                  axis=1)
        table[n - i - 1, :] = np.tile(base, (1, int(p ** i)))

    table = table.astype("int64") - 1

    s = 1 / (np.sqrt(np.pi) ** n)

    wm = s * np.prod(w_1d[table], axis=0)
    xi = np.sqrt(2) * hermite_roots[table]

    return wm, wm, xi


def _hermite_coeff(order: int) -> List:
    """ Give the 0 to p-th order physician Hermite polynomial coefficients, where p is the
    order from the argument. The returned coefficients is ordered from highest to lowest.
    Also note that this implementation is different from the np.hermite method.
    Parameters
    ----------
    order:  int
        The order of Hermite polynomial
    Returns
    -------
    H: List
        The 0 to p-th order Hermite polynomial coefficients in a list.
    """
    H0 = np.array([1])
    H1 = np.array([2, 0])

    H = [H0, H1]

    for i in range(2, order + 1):
        H.append(2 * np.append(H[i - 1], 0) -
                 2 * (i - 1) * np.pad(H[i - 2], (2, 0), 'constant', constant_values=0))

    return H
