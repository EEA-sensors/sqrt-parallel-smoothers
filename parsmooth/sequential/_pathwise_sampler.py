from typing import Optional, Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlag

from parsmooth._base import MVNStandard, MVNSqrt, are_inputs_compatible, FunctionalModel
from parsmooth._utils import tria, none_or_shift, none_or_concat


def sampler(key: jnp.ndarray, n_samples: int, transition_model: FunctionalModel,
            filter_trajectory: MVNSqrt or MVNStandard,
            linearization_method: Callable,
            nominal_trajectory: Optional[MVNStandard or MVNSqrt],
            batch_axis=-1):
    keys = jax.random.split(key, n_samples)
    vmapped_sampler = jax.vmap(lambda k: _sampler(k,
                                                  transition_model,
                                                  filter_trajectory,
                                                  linearization_method,
                                                  nominal_trajectory),
                               in_axes=[0],
                               out_axes=batch_axis)
    samples = vmapped_sampler(keys)
    return samples


def _sample_mvn(key: jnp.ndarray, x: MVNStandard or MVNSqrt):
    if isinstance(x, MVNStandard):
        return jax.random.multivariate_normal(key, *x)
    elif isinstance(x, MVNSqrt):
        m, L = x
        eps = jax.random.normal(key, x.mean.shape)
        return m + L @ eps


def _sampler(key: jnp.ndarray, transition_model: FunctionalModel,
             filter_trajectory: MVNSqrt or MVNStandard,
             linearization_method: Callable, nominal_trajectory: Optional[MVNStandard or MVNSqrt]):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory)
    T = filter_trajectory.mean.shape[0]
    keys = jax.random.split(key, T)

    are_inputs_compatible(filter_trajectory, nominal_trajectory)

    def sample_one(F_x, cov_or_chol, b, xf, xs, op_key):
        if isinstance(xf, MVNSqrt):
            return _sqrt_sample(F_x, cov_or_chol, b, xf, xs, op_key)
        return _standard_sample(F_x, cov_or_chol, b, xf, xs, op_key)

    def body(smoothed, inputs):
        filtered, ref, op_key = inputs
        F_x, cov_or_chol, b = linearization_method(transition_model, ref)
        smoothed_state = sample_one(F_x, cov_or_chol, b, filtered, smoothed, op_key)

        return smoothed_state, smoothed_state

    last_state_sample = _sample_mvn(keys[0], last_state)

    _, sampled_states = jax.lax.scan(body,
                                     last_state_sample,
                                     [none_or_shift(filter_trajectory, -1), none_or_shift(nominal_trajectory, -1),
                                      keys[1:]],
                                     reverse=True)

    sampled_states = none_or_concat(sampled_states, last_state_sample, -1)
    return sampled_states


def _standard_sample(F, Q, b, xf, xs, key):
    mf, Pf = xf
    S = F @ Pf @ F.T + Q
    gain = Pf @ jlag.solve(S, F, sym_pos=True).T

    inc_Sig = Pf - gain @ S @ gain.T
    inc_m = mf - gain @ (F @ mf + b)

    inc = _sample_mvn(key, MVNStandard(inc_m, inc_Sig))
    return gain @ xs + inc


def _sqrt_sample(F, cholQ, b, xf, xs, key):
    mf, cholPf = xf

    nx = F.shape[0]
    Phi = jnp.block([[F @ cholPf, cholQ],
                     [cholPf, jnp.zeros_like(F)]])
    tria_Phi = tria(Phi)
    Phi11 = tria_Phi[:nx, :nx]
    Phi21 = tria_Phi[nx:, :nx]
    Phi22 = tria_Phi[nx:, nx:]

    gain = jlag.solve_triangular(Phi11, Phi21.T, trans=True, lower=True).T
    inc_L = tria(jnp.concatenate([Phi22, gain @ Phi21], axis=1))
    inc_m = mf - gain @ (F @ mf + b)

    inc = _sample_mvn(key, MVNSqrt(inc_m, inc_L))
    return gain @ xs + inc
