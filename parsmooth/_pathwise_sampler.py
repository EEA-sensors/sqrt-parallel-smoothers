from functools import partial
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlag

from parsmooth._base import MVNStandard, MVNSqrt, FunctionalModel
from parsmooth._utils import tria, none_or_shift, none_or_concat


def _par_sampling(key: jnp.ndarray,
                  n_samples: int,
                  transition_model: FunctionalModel,
                  filter_trajectory: MVNSqrt or MVNStandard,
                  linearization_method: Callable,
                  nominal_trajectory: Union[MVNSqrt, MVNStandard]):
    gains, incs, last_state_sample = _sampling_common(key, n_samples, transition_model, filter_trajectory,
                                                      linearization_method, nominal_trajectory)

    @jax.vmap
    def operator(elem1, elem2):
        G1, e1 = elem1
        G2, e2 = elem2

        G = G2 @ G1
        e = jnp.einsum("ij,...j->...i", G2, e1) + e2
        return G, e

    Gs = none_or_concat(gains, jnp.zeros_like(gains[0]), -1)
    es = none_or_concat(incs, last_state_sample, -1)

    _, sampled_states = jax.lax.associative_scan(operator,
                                                 [Gs, es],
                                                 reverse=True)

    return sampled_states


def _seq_sampling(key: jnp.ndarray,
                  n_samples: int,
                  transition_model: FunctionalModel,
                  filter_trajectory: MVNSqrt or MVNStandard,
                  linearization_method: Callable,
                  nominal_trajectory: Union[MVNSqrt, MVNStandard]):
    gains, incs, last_state_sample = _sampling_common(key, n_samples, transition_model, filter_trajectory,
                                                      linearization_method, nominal_trajectory)

    def body(prev_xs, inputs):
        gain, inc = inputs
        xs = jnp.einsum("ij,...j->...i", gain, prev_xs) + inc
        return xs, xs

    _, sampled_states = jax.lax.scan(body,
                                     last_state_sample,
                                     [gains, incs],
                                     reverse=True)

    sampled_states = none_or_concat(sampled_states, last_state_sample, -1)
    return sampled_states


def _sampling_common(key: jnp.ndarray,
                     n_samples: int,
                     transition_model: FunctionalModel,
                     filter_trajectory: MVNSqrt or MVNStandard,
                     linearization_method: Callable,
                     nominal_trajectory: Union[MVNSqrt, MVNStandard]):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory)
    filter_trajectory = none_or_shift(filter_trajectory, -1)
    F_x, cov_or_chol, b = jax.vmap(linearization_method, in_axes=[None, 0])(transition_model,
                                                                            none_or_shift(nominal_trajectory, -1))
    T = F_x.shape[0]
    eps = jax.random.normal(key, (T + 1, n_samples) + b.shape[1:])

    if isinstance(filter_trajectory, MVNSqrt):
        gain, inc = jax.vmap(_sqrt_gain_and_inc)(F_x, cov_or_chol, b, filter_trajectory, eps[1:])
    else:
        gain, inc = jax.vmap(_standard_gain_and_inc)(F_x, cov_or_chol, b, filter_trajectory, eps[1:])

    last_state_sample = _make_mvn(last_state, eps[0])

    return gain, inc, last_state_sample


@partial(jax.vmap, in_axes=[None, 0])
def _make_mvn(x: Union[MVNSqrt, MVNStandard], eps: jnp.ndarray):
    m, cov_or_chol = x
    if isinstance(x, MVNStandard):
        L = jnp.linalg.cholesky(cov_or_chol)
    elif isinstance(x, MVNSqrt):
        L = cov_or_chol
    else:
        raise ValueError(f"`x` must be a MVNStandard or a MVNSqrt. {type(x)} was passed")
    return m + L @ eps


def _standard_gain_and_inc(F, Q, b, xf, eps):
    mf, Pf = xf

    S = F @ Pf @ F.T + Q
    gain = Pf @ jlag.solve(S, F, sym_pos=True).T

    inc_Sig = Pf - gain @ S @ gain.T
    inc_m = mf - gain @ (F @ mf + b)

    inc = _make_mvn(MVNStandard(inc_m, inc_Sig), eps)
    return gain, inc


def _sqrt_gain_and_inc(F, cholQ, b, xf, eps):
    mf, cholPf = xf

    nx = F.shape[0]
    Phi = jnp.block([[F @ cholPf, cholQ],
                     [cholPf, jnp.zeros_like(F)]])
    tria_Phi = tria(Phi)
    Phi11 = tria_Phi[:nx, :nx]
    Phi21 = tria_Phi[nx:, :nx]
    inc_L = tria_Phi[nx:, nx:]

    gain = jlag.solve_triangular(Phi11, Phi21.T, trans=True, lower=True).T
    inc_m = mf - gain @ (F @ mf + b)

    inc = _make_mvn(MVNSqrt(inc_m, inc_L), eps)
    return gain, inc
