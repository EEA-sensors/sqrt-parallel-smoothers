from functools import partial

import jax.numpy as jnp
from jax import random
from jax.lax import scan


def _transition_function(x):
    return jnp.log(44.7) + x - jnp.exp(x)


def _mean_transition_function(x):
    return jnp.log(44.7) + x - jnp.exp(x)


def _cov_transition_function(x, Q):
    return Q

def _chol_transition_function(x, Q):
    return jnp.sqrt(Q)


def _observation_function(x, lam, seed):
    key = random.PRNGKey(seed)
    return random.poisson(key, lam * jnp.exp(x), shape=x.shape).astype(float)


def _mean_observation_function(x, lam):
    return lam * jnp.exp(x)


def _cov_observation_function(x, lam):
    return (lam * jnp.exp(x)).reshape(1, 1)


def _chol_observation_function(x, lam):
    return (jnp.sqrt(lam * jnp.exp(x))).reshape(1, 1)


def make_parameters(lam, seed, Q):

    mean_transition_function = partial(_mean_transition_function)
    cov_transition_function = partial(_cov_transition_function, Q=Q)
    chol_transition_function = partial(_chol_transition_function, Q=Q)

    mean_observation_function = partial(_mean_observation_function, lam=lam)
    cov_observation_function = partial(_cov_observation_function, lam=lam)
    chol_observation_function = partial(_chol_observation_function, lam=lam)

    return mean_transition_function, cov_transition_function, mean_observation_function, cov_observation_function, chol_transition_function, chol_observation_function


def get_data(x0: jnp.ndarray, dt: float, T: int, Q: jnp.ndarray, lam: jnp.ndarray, seed: int):
    ts = jnp.linspace(dt, (T + 1) * dt, T)
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
