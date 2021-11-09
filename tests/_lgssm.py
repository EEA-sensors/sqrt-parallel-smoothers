import jax.numpy as jnp
import numpy as np


def transition_function(x, A):
    """ Deterministic transition function used in the state space model
    Parameters
    ----------
    x: array_like
        The current state
    q: array_like
        The noise param
    A: array_like
        transition matrix
    Returns
    -------
    out: array_like
        The transitioned state
    """
    return jnp.dot(A, x)


def observation_function(x, H):
    """
    Returns the observed angles as function of the state and the sensors locations
    Parameters
    ----------
    x: array_like
        The current state
    r: array_like
        The error param
    H: array_like
        observation matrix
    Returns
    -------
    y: array_like
        The observed data
    """
    return jnp.dot(H, x)


def get_data(x0, A, H, R, Q, b, c, T, random_state=None, chol_R=None):
    """
    Parameters
    ----------
    x0: array_like
        true initial state
    A: array_like
        transition matrix
    H: array_like
        transition matrix
    R: array_like
        observation model covariance
    Q: array_like
        noise covariance
    b: array_like
        transition offset
    c: array_like
        observation offset
    T: int
        number of time steps
    random_state: np.random.RandomState or int, optional
        numpy random state
    chol_R: array_like, optional
        cholesky of R

    Returns
    -------
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    R_shape = R.shape[0]
    Q_shape = Q.shape[0]
    normals = random_state.randn(T, Q_shape + R_shape).astype(np.float32)
    if chol_R is None:
        chol_R = np.linalg.cholesky(R)
    chol_Q = np.linalg.cholesky(Q)

    x = np.copy(x0).astype(np.float32)
    observations = np.empty((T, R_shape), dtype=np.float32)
    true_states = np.empty((T + 1, Q_shape), dtype=np.float32)
    true_states[0] = x

    for i in range(T):
        x = A @ x + chol_Q @ normals[i, :Q_shape] + b
        true_states[i + 1] = x
        y = H @ x + chol_R @ normals[i, Q_shape:] + c
        observations[i] = y

    return true_states, observations
