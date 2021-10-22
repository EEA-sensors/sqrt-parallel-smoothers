from typing import Optional, Callable, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlag

from parsmooth._base import MVNStandard, MVNSqrt, are_inputs_compatible, FunctionalModel, ConditionalMomentsModel
from parsmooth._utils import tria, none_or_shift, none_or_concat


def smoothing(transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              filter_trajectory: Union[MVNSqrt, MVNStandard],
              linearization_method: Callable,
              nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory)

    if nominal_trajectory is not None:
        are_inputs_compatible(filter_trajectory, nominal_trajectory)

    def smooth_one(F_x, cov_or_chol, b, xf, xs):
        are_inputs_compatible(xf, xs)
        if isinstance(xf, MVNSqrt):
            return _sqrt_smooth(F_x, cov_or_chol, b, xf, xs)
        return _standard_smooth(F_x, cov_or_chol, b, xf, xs)

    def body(smoothed, inputs):
        filtered, ref = inputs
        if ref is None:
            ref = smoothed
        F_x, cov_or_chol, b = linearization_method(transition_model, ref)
        smoothed_state = smooth_one(F_x, cov_or_chol, b, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_state,
                                      [none_or_shift(filter_trajectory, -1), none_or_shift(nominal_trajectory, -1)],
                                      reverse=True)

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


def _standard_smooth(F, Q, b, xf, xs):
    mf, Pf = xf
    ms, Ps = xs

    mean_diff = ms - (b + F @ mf)
    S = F @ Pf @ F.T + Q
    cov_diff = Ps - S

    gain = Pf @ jlag.solve(S, F, sym_pos=True).T
    ms = mf + gain @ mean_diff
    Ps = Pf + gain @ cov_diff @ gain.T

    return MVNStandard(ms, Ps)


def _sqrt_smooth(F, cholQ, b, xf, xs):
    mf, cholPf = xf
    ms, cholPs = xs

    nx = F.shape[0]
    Phi = jnp.block([[F @ cholPf, cholQ],
                     [cholPf, jnp.zeros_like(F)]])
    tria_Phi = tria(Phi)
    Phi11 = tria_Phi[:nx, :nx]
    Phi21 = tria_Phi[nx:, :nx]
    Phi22 = tria_Phi[nx:, nx:]
    gain = jlag.solve_triangular(Phi11, Phi21.T, trans=True, lower=True).T

    mean_diff = ms - (b + F @ mf)
    mean = mf + gain @ mean_diff
    chol = tria(jnp.concatenate([Phi22, gain @ cholPs], axis=1))

    return MVNSqrt(mean, chol)
