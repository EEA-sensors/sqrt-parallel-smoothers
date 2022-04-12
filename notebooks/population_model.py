from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.lax import scan


def _transition_function(x):
    """ Deterministic transition function used in the state space model
    Parameters
    ----------
    x: array_like
        The current state
    Returns
    -------
    out: array_like
        The transitioned state
    """
    return jnp.log(44.7) + x - jnp.exp(x)


def _mean_transition_function(x):
    """ Returns the expected value of the transition function
    Parameters
    ----------
    x: array_like
        The current state
    Returns
    -------
    out: array_like
        The expected value of the transition function
    """
    return jnp.log(44.7) + x - jnp.exp(x)


def _cov_transition_function(_x, Q):
    """ Covariance of process noise also can be interpreted as environmental noise
    Parameters
    ----------
    Q: array_like
        covariance of process noise
    Returns
    -------
    out: array_like
        The transition covariance
    """
    return Q


def _chol_transition_function(_x, Q):
    """ Square-root of covariance of process noise
    Parameters
    ----------
    Q: array_like
        covariance of process noise
    Returns
    -------
    out: array_like
        The cholesky transition covariance
    """
    return jnp.sqrt(Q)


def _observation_function(x, lam, key):
    """
    Returns the noisily observed Ricker map
    Parameters
    ----------
    x: array_like
        The current state
    lam: float
        The poisson parameter
    key: jnp.ndarray
        jax random key
    Returns
    -------
    y: array_like

    """
    return random.poisson(key, lam * jnp.exp(x), shape=x.shape).astype(float)


def _mean_observation_function(x, lam):
    """ Returns the expected value of the observation function
     Parameters
     ----------
     x: array_like
         The current state
    lam: array_like
         The poisson parameter
     Returns
     -------
     out: array_like
         The expected value of the observation function
     """
    return lam * jnp.exp(x)


def _cov_observation_function(x, lam):
    """ Covariance of observation noise
    Parameters
    ----------
    x: array_like
        The current state
    lam: array_like
        The Poisson parameter
    Returns
    -------
    out: callable
        The observation covariance function
    """
    return (lam * jnp.exp(x)).reshape(1, 1)


def _chol_observation_function(x, lam):
    """ Square-root of covariance of observation noise
    Parameters
    ----------
    x: array_like
        The current state
    lam: array_like
        The Poisson parameter
    Returns
    -------
    out: callable
        The observation cholesky function
    """
    return (jnp.sqrt(lam * jnp.exp(x))).reshape(1, 1)


def make_parameters(lam, Q):
    """ Paremeters of the stochastic Ricker map model which is explained in
    [1] "A comparison of inferential methods for highly nonlinear state space models in ecology and epidemiology
    Fasiolo, Matteo and Pya, Natalya and Wood, Simon N"
    [2] "Iterative filtering and smoothing in nonlinear and non-Gaussian systems using conditional moments,
    Tronarp, Filip and Garcia-Fernandez, Angel F and Särkkä, Simo"

        Parameters
        ----------
        lam: float
            The poisson parameter
        Q: array_like
            Transition covariance
        Returns
        -------
        mean_transition_function: callable
            The conditional mean of transition function
        mean_observation_function: callable
            The conditional mean of observation function
        cov_transition_function: callable
            The transition conditional covariance function
        cov_observation_function: callable
            The observation conditional covariance function
        chol_transition_function: callable
            The transition cholesky of conditional covariance function
        cov_observation_function: callable
            The observation cholesky of conditional covariance function
        """

    mean_transition_function = partial(_mean_transition_function)
    cov_transition_function = partial(_cov_transition_function, Q=Q)
    chol_transition_function = partial(_chol_transition_function, Q=Q)

    mean_observation_function = partial(_mean_observation_function, lam=lam)
    cov_observation_function = partial(_cov_observation_function, lam=lam)
    chol_observation_function = partial(_chol_observation_function, lam=lam)

    return mean_transition_function, cov_transition_function, mean_observation_function, cov_observation_function, chol_transition_function, chol_observation_function


def get_data(x0: jnp.ndarray, T: int, Q: jnp.ndarray, lam: jnp.ndarray, key: jnp.ndarray):
    """
    Parameters
    ----------
    x0: array_like
        initial state
    T: int
        number of time steps
    Q: array_like
        covariance of transition noise
    lam: float
        poisson parameter
    key: jnp.ndarray
        jax random key
    Returns
    -------
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """
    key, gaussian_key = jax.random.split(key)
    
    chol_Q = jnp.linalg.cholesky(Q)
    noises = jax.random.normal(gaussian_key, shape=(T, x0.shape[0])) @ chol_Q.T

    def body(x_k, inputs):
        noise, poisson_key = inputs
        x_k_p1 = _transition_function(x_k) + noise
        y_k = _observation_function(x_k_p1, lam, poisson_key)
        return x_k_p1, (x_k_p1, y_k)

    poisson_keys = jax.random.split(key, T)

    _, (true_states, observations) = scan(body, x0, (noises, poisson_keys))
    true_states = jnp.insert(true_states, 0, x0, 0)
    return true_states, observations
