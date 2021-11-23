from functools import partial

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


def _observation_function(x, lam, seed):
    """
    Returns the noisily observed Ricker map
    Parameters
    ----------
    x: array_like
        The current state
    lam: float
        The poisson parameter
    seed: int
        The random seed
    Returns
    -------
    y: array_like

    """
    key = random.PRNGKey(seed)
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
    [1] "A comparison of inferential methods for highly nonlinear state space models
    in ecology and epidemiology
    Fasiolo, Matteo and Pya, Natalya and Wood, Simon N"
    [2] "Iterative filtering and smoothing in nonlinear and non-Gaussian systems using conditional moments,
    Tronarp, Filip and Garcia-Fernandez, Angel F and Särkkä, Simo},

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


def get_data(x0: jnp.ndarray, T: int, Q: jnp.ndarray, lam: jnp.ndarray, seed: int):
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
    seed: int
        numpy seed
    Returns
    -------
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """
    key = random.PRNGKey(seed)

    noises = (jnp.linalg.cholesky(Q) * random.normal(key, shape=(T, x0.shape[0])))

    def body(carry, noise):
        x_k = carry[0]
        x_k_p1 = _transition_function(x_k) + noise
        y_k = _observation_function(x_k, lam, seed)

        carry = (x_k_p1, y_k)
        return carry, (x_k_p1, y_k)

    _, output = scan(body, (x0, x0 * 0.), noises)
    true_states = jnp.squeeze(output[0], axis=1)
    observations = output[1]

    return true_states, observations
