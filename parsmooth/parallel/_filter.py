from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth.parallel._operators import filtering_operator
from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, are_inputs_compatible
from parsmooth._utils import tria


def filtering(observations: jnp.ndarray,
              x0: MVNStandard or MVNSqrt,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method: Callable,
              nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None):

    n_observations = observations.shape[0]

    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    def make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i):
        if isinstance(x0, MVNSqrt):
            return _sqrt_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i)
        return _standard_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i)

    @jax.vmap
    def make_params(obs, i):
        return make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, obs, i)

    associative_params = make_params(observations, jnp.arange(n_observations))
    _, filtered_means, filtered_covariances, _, _ = jax.lax.associative_scan(filtering_operator, *associative_params)

    return jax.vmap(MVNParams)(filtered_means, filtered_covariances)


def _standard_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i):
    predicate = i == 0

    def _first(_):
        return _standard_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y)

    def _generic(_):
        return _standard_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y)

    return jax.lax.cond(predicate,
                    _first,
                    _generic,
                    None)


def _standard_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y):

    F, Q, b = linearization_method(transition_model, nominal_trajectory)
    H, R, c = linearization_method(observation_model, nominal_trajectory)

    m1 = F @ x0.mean + b
    P1 = F @ x0.cov @ F.T + Q

    S = H @ P1 @ H.T + R
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ P1).T
    A = jnp.zeros(F.shape)

    b_std = m1 + K @ (y - H @ m1 - c)
    C = P1 - (K @ S @ K.T)

    temp = (S_invH @ F).T
    eta = temp @ (y - H @ b - c)
    J = temp @ H @ F

    return A, b_std, C, eta, J


def _standard_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y):

    F, Q, b = linearization_method(transition_model, nominal_trajectory)
    H, R, c = linearization_method(observation_model, nominal_trajectory)

    S = H @ Q @ H.T + R
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ Q).T
    A = F - K @ H @ F
    b_std = b + K @ (y - H @ b - c)
    C = Q - K @ H @ Q

    temp = (S_invH @ F).T
    eta = temp @ (y - H @ b - c)
    J = temp @ H @ F

    return A, b_std, C, eta, J


def _sqrt_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i):

    predicate = i == 0

    def _first(_):
        return _sqrt_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y)

    def _generic(_):
        return _sqrt_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y)

    return jax.lax.cond(predicate,
                    _first,
                    _generic,
                    None)


def _sqrt_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y):

    F, cholQ, b = linearization_method(transition_model, nominal_trajectory)
    H, cholR, c = linearization_method(observation_model, nominal_trajectory)

    nx = cholQ.shape[0]
    ny = cholR.shape[0]

    m1 = F @ x0.mean + b
    N1_ = tria(jnp.concatenate((F @ x0.chol, cholQ), axis=1))
    Psi_ = jnp.block([[H @ N1_, cholR], [N1_, jnp.zeros((N1_.shape[0], cholR.shape[1]))]])
    Tria_Psi_ = tria(Psi_)
    Psi11_ = Tria_Psi_[:ny, :ny]
    Psi21_ = Tria_Psi_[ny: ny + nx, :ny]
    Psi22_ = Tria_Psi_[ny: ny + nx, ny:]
    Y1 = Psi11_
    K1 = jlinalg.solve(Psi11_.T, Psi21_.T).T

    A = jnp.zeros_like(F)
    b_sqr = m1 + K1 @ (y - H @ m1 - c)
    U = Psi22_

    Z1 = jlinalg.solve(Y1, H @ F).T
    eta = jlinalg.solve(Y1.T, Z1.T).T @ (y - H @ b - c)
    if nx > ny:
        Z = jnp.block([Z1, jnp.zeros((nx, nx - ny))])
    else:
        Z = jnp.block([Z1, jnp.zeros((nx, ny - nx))])

    return A, b_sqr, U, eta, Z


def _sqrt_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y):

    F, cholQ, b = linearization_method(transition_model, nominal_trajectory)
    H, cholR, c = linearization_method(observation_model, nominal_trajectory)

    nx = cholQ.shape[0]
    ny = cholR.shape[0]
    Psi = jnp.block([[H @ cholQ, cholR], [cholQ, jnp.zeros((nx, ny))]])
    Tria_Psi = tria(Psi)
    Psi11 = Tria_Psi[:ny, :ny]
    Psi21 = Tria_Psi[ny:ny + nx, :ny]
    Psi22 = Tria_Psi[ny:ny + nx, ny:]
    Y = Psi11
    K = jlinalg.solve(Psi11.T, Psi21.T).T
    A = F - K @ H @ F
    b_sqr = b + K @ (y - H @ b - c)
    U = Psi22

    Z1 = jlinalg.solve(Y, H @ F).T
    eta = jlinalg.solve(Y.T, Z1.T).T @ (y - H @ b - c)

    if nx > ny:
        Z = jnp.block([Z1, jnp.zeros((nx, nx - ny))])
    else:
        Z = jnp.block([Z1, jnp.zeros((nx, ny - nx))])

    return A, b_sqr, U, eta, Z

