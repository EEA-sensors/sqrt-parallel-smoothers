from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, are_inputs_compatible, ConditionalMomentsModel
from parsmooth._utils import tria, none_or_concat, mvn_loglikelihood
from parsmooth.parallel._operators import sqrt_filtering_operator, standard_filtering_operator


def filtering(observations: jnp.ndarray,
              x0: Union[MVNSqrt, MVNStandard],
              transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              observation_model: Union[FunctionalModel, ConditionalMomentsModel],
              linearization_method: Callable,
              nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
              return_loglikelihood=False):
    T = observations.shape[0]
    m0, chol_or_cov_0 = x0
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    else:
        nominal_mean = jnp.zeros_like(m0, shape=(T + 1,) + m0.shape)
        nominal_cov_or_chol = jnp.repeat(jnp.eye(m0.shape[-1])[None, ...], T + 1, 0)
        nominal_trajectory = type(x0)(nominal_mean, nominal_cov_or_chol)  # this is kind of a hack but I've seen worse.

    if isinstance(x0, MVNSqrt):
        associative_params, linearized_ssm = _sqrt_associative_params(linearization_method, transition_model,
                                                                      observation_model,
                                                                      nominal_trajectory, x0, observations)
        _, filtered_means, filtered_chol_or_cov, _, _ = jax.lax.associative_scan(jax.vmap(sqrt_filtering_operator),
                                                                                 associative_params)


    else:
        associative_params, linearized_ssm = _standard_associative_params(linearization_method, transition_model,
                                                                          observation_model,
                                                                          nominal_trajectory, x0, observations)
        _, filtered_means, filtered_chol_or_cov, _, _ = jax.lax.associative_scan(jax.vmap(standard_filtering_operator),
                                                                                 associative_params)

    filtered_means = none_or_concat(filtered_means, m0, position=1)
    filtered_chol_or_cov = none_or_concat(filtered_chol_or_cov, chol_or_cov_0, position=1)

    if isinstance(x0, MVNSqrt):
        res = jax.vmap(MVNSqrt)(filtered_means, filtered_chol_or_cov)
    else:
        res = jax.vmap(MVNStandard)(filtered_means, filtered_chol_or_cov)

    if return_loglikelihood:
        if isinstance(x0, MVNSqrt):
            ells = jax.vmap(_sqrt_loglikelihood)(*linearized_ssm, filtered_means[:-1], filtered_chol_or_cov[:-1],
                                                 observations)
        else:
            ells = jax.vmap(_standard_loglikelihood)(*linearized_ssm, filtered_means[:-1], filtered_chol_or_cov[:-1],
                                                     observations)
        return res, jnp.sum(ells)
    return res


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

    return (A, b_std, C, eta, J), (F, Q, b, H, R, c)


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

    return (A, b_sqr, U, eta, Z), (F, cholQ, b, H, cholR, c)


def _sqrt_loglikelihood(F, cholQ, b, H, cholR, c, m_t_1, cholP_t_1, y_t):
    predicted_mean = F @ m_t_1 + b
    predicted_chol = tria(jnp.concatenate([F @ cholP_t_1, cholQ], axis=1))
    obs_mean = H @ predicted_mean + c
    obs_chol = tria(jnp.concatenate([H @ predicted_chol, cholR], axis=1))
    return mvn_loglikelihood(y_t - obs_mean, obs_chol)


def _standard_loglikelihood(F, Q, b, H, R, c, m_t_1, P_t_1, y_t):
    predicted_mean = F @ m_t_1 + b
    predicted_cov = F @ P_t_1 @ F.T + Q
    obs_mean = H @ predicted_mean + c
    obs_cov = H @ predicted_cov @ H.T + R
    chol = jnp.linalg.cholesky(obs_cov)
    return mvn_loglikelihood(y_t - obs_mean, chol)
