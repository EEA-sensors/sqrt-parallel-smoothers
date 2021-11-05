from functools import partial

import jax.numpy as jnp
from jax import lax, jit


def _transition_function(x, dt):
    """ Deterministic transition function used in the state space model

    Parameters
    ----------
    x: array_like
        The current state
    q: array_like
        realisation of the transition noise
    dt: float
        Time step between observations

    Returns
    -------
    out: array_like
        The transitioned state
    """
    w = x[-1]
    predicate = jnp.abs(w) < 1e-6

    coswt = jnp.cos(w * dt)
    sinwt = jnp.sin(w * dt)

    def true_fun(_):
        return coswt, 0., sinwt, dt

    def false_fun(_):
        coswto = coswt - 1
        return coswt, coswto / w, sinwt, sinwt / w

    coswt, coswtopw, sinwt, sinwtpw = lax.cond(predicate, true_fun, false_fun, None)

    F = jnp.array([[1, 0, sinwtpw, -coswtopw, 0],
                   [0, 1, coswtopw, sinwtpw, 0],
                   [0, 0, coswt, sinwt, 0],
                   [0, 0, -sinwt, coswt, 0],
                   [0, 0, 0, 0, 1]])
    return F @ x


def _observation_function(x, s1, s2):
    """
    Returns the observed angles as function of the state and the sensors locations

    Parameters
    ----------
    x: array_like
        The current state
    r: array_like
        realisation of the observation error
    s1: array_like
        The first sensor location
    s2: array_like
        The second sensor location

    Returns
    -------
    y: array_like
        The observed angles, the first component is the angle w.r.t. the first sensor, the second w.r.t the second.
    """
    temp = jnp.array([jnp.arctan2(x[1] - s1[1], x[0] - s1[0]),
                      jnp.arctan2(x[1] - s2[1], x[0] - s2[0])])
    return temp


def make_parameters(qc, qw, r, dt, s1, s2):
    """ Discretizes the model with continuous transition noise qc, for step-size dt.
    The model is described in "Multitarget-multisensor tracking: principles and techniques" by
    Bar-Shalom, Yaakov and Li, Xiao-Rong

    Parameters
    ----------
    qc: float
        Transition covariance of the continuous SSM
    qw: float
        Transition covariance of the continuous SSM
    r: float
        Observation error standard deviation
    dt: float
        Discretization time step
    s1: array_like
        The location of the first sensor
    s2: array_like
        The location of the second sensor

    Returns
    -------
    Q: array_like
        The transition covariance matrix for the discrete SSM
    R: array_like
        The observation covariance matrix
    observation_function: callable
        The observation function
    transition_function: callable
        The transition function
    """

    Q = jnp.array([[qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0, 0],
                   [0, qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0],
                   [qc * dt ** 2 / 2, 0, qc * dt, 0, 0],
                   [0, qc * dt ** 2 / 2, 0, qc * dt, 0],
                   [0, 0, 0, 0, dt * qw]])

    R = r ** 2 * jnp.eye(2)

    observation_function = jit(partial(_observation_function, s1=s1, s2=s2))
    transition_function = jit(partial(_transition_function, dt=dt))

    return Q, R, observation_function, transition_function
