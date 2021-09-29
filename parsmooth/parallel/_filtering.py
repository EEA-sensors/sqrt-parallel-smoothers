from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, are_inputs_compatible
from parsmooth._utils import tria, none_or_concat
from parsmooth.parallel._operators import sqrt_filtering_operator, standard_filtering_operator


def filtering(observations: jnp.ndarray,
              x0: MVNStandard or MVNSqrt,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method: Callable,
              nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None):
    T = observations.shape[0]
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    else:
        m0, chol_or_cov_0 = x0
        nominal_mean = jnp.zeros_like(m0, shape=(T + 1,) + m0.shape)
        nominal_cov_or_chol = jnp.repeat(jnp.eye(m0.shape[-1])[None, ...], T + 1, 0)
        nominal_trajectory = type(x0)(nominal_mean, nominal_cov_or_chol)  # this is kind of a hack but I've seen worse.

    if isinstance(x0, MVNSqrt):
        associative_params = _sqrt_associative_params(linearization_method, transition_model, observation_model,
                                                      nominal_trajectory, x0, observations)
        _, filtered_means, filtered_chol, _, _ = jax.lax.associative_scan(jax.vmap(sqrt_filtering_operator),
                                                                          associative_params)
        res = jax.vmap(MVNSqrt)(filtered_means, filtered_chol)

    else:
        associative_params = _standard_associative_params(linearization_method, transition_model, observation_model,
                                                          nominal_trajectory, x0, observations)
        _, filtered_means, filtered_cov, _, _ = jax.lax.associative_scan(jax.vmap(standard_filtering_operator),
                                                                         associative_params)
        res = jax.vmap(MVNStandard)(filtered_means, filtered_cov)

    return none_or_concat(res, x0, position=1)


def _standard_associative_params(linearization_method, transition_model, observation_model,
                                 nominal_trajectory, x0, ys):
    T = ys.shape[0]
    n_k_1 = jax.tree_map(lambda z: z[:-1], nominal_trajectory)
    n_k = jax.tree_map(lambda z: z[1:], nominal_trajectory)

    m0, P0 = x0
    ms = jnp.concatenate([m0[None, ...], jnp.zeros_like(m0, shape=(T - 1,) + m0.shape)])
    Ps = jnp.concatenate([P0[None, ...], jnp.zeros_like(P0, shape=(T - 1,) + P0.shape)])

    vmapped_fn = jax.vmap(_standard_associative_params_one, in_axes=[None, None, None, 0, 0, 0, 0, 0])
    return vmapped_fn(linearization_method, transition_model, observation_model, n_k_1, n_k, ms, Ps, ys)


def _standard_associative_params_one(linearization_method, transition_model, observation_model, n_k_1, n_k, m, P, y):
    F, Q, b = linearization_method(transition_model, n_k_1)
    H, R, c = linearization_method(observation_model, n_k)

    m = F @ m + b
    P = F @ P @ F.T + Q

    S = H @ P @ H.T + R
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ P).T
    A = F - K @ H @ F

    b_std = m + K @ (y - H @ m - c)
    C = P - (K @ S @ K.T)

    temp = (S_invH @ F).T
    eta = temp @ (y - H @ b - c)
    J = temp @ H @ F

    return A, b_std, C, eta, J


def _sqrt_associative_params(linearization_method, transition_model, observation_model,
                             nominal_trajectory, x0, ys):
    T = ys.shape[0]
    n_k_1 = jax.tree_map(lambda z: z[:-1], nominal_trajectory)
    n_k = jax.tree_map(lambda z: z[1:], nominal_trajectory)

    m0, L0 = x0
    ms = jnp.concatenate([m0[None, ...], jnp.zeros_like(m0, shape=(T - 1,) + m0.shape)])
    Ls = jnp.concatenate([L0[None, ...], jnp.zeros_like(L0, shape=(T - 1,) + L0.shape)])

    vmapped_fn = jax.vmap(_sqrt_associative_params_one, in_axes=[None, None, None, 0, 0, 0, 0, 0])

    return vmapped_fn(linearization_method, transition_model, observation_model, n_k_1, n_k, ms, Ls, ys)


def _sqrt_associative_params_one(linearization_method, transition_model, observation_model,
                                 n_k_1, n_k, m0, L0, y):
    F, cholQ, b = linearization_method(transition_model, n_k_1)
    H, cholR, c = linearization_method(observation_model, n_k)

    nx = cholQ.shape[0]
    ny = cholR.shape[0]

    m1 = F @ m0 + b
    N1_ = tria(jnp.concatenate((F @ L0, cholQ), axis=1))
    Psi_ = jnp.block([[H @ N1_, cholR],
                      [N1_, jnp.zeros((N1_.shape[0], cholR.shape[1]))]])
    Tria_Psi_ = tria(Psi_)
    Psi11 = Tria_Psi_[:ny, :ny]
    Psi21 = Tria_Psi_[ny: ny + nx, :ny]
    U = Tria_Psi_[ny: ny + nx, ny:]

    K = jlinalg.solve_triangular(Psi11, Psi21.T, trans=True, lower=True).T

    A = F - K @ H @ F
    b_sqr = m1 + K @ (y - H @ m1 - c)

    Z = jlinalg.solve_triangular(Psi11, H @ F, lower=True).T
    eta = jlinalg.solve_triangular(Psi11, Z.T, trans=True, lower=True).T @ (y - H @ b - c)

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    return A, b_sqr, U, eta, Z
