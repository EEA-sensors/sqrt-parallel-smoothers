from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, are_inputs_compatible, ConditionalMomentsModel
from parsmooth._utils import none_or_concat, tria
from parsmooth.parallel._operators import sqrt_smoothing_operator, \
    standard_smoothing_operator


def smoothing(transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              filter_trajectory: Union[MVNSqrt, MVNStandard],
              linearization_method: Callable,
              nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None):
    if nominal_trajectory is not None:
        are_inputs_compatible(filter_trajectory, nominal_trajectory)

    else:
        ms, chol_or_covs = filter_trajectory
        T, nx = ms.shape
        nominal_mean = jnp.zeros_like(ms)
        nominal_cov_or_chol = jnp.repeat(jnp.eye(nx)[None, ...], T, 0)

        # this is kind of a hack but I've seen worse.
        nominal_trajectory = type(filter_trajectory)(nominal_mean, nominal_cov_or_chol)

    if isinstance(filter_trajectory, MVNSqrt):
        associative_params = _associative_params(linearization_method, transition_model,
                                                 nominal_trajectory, filter_trajectory, True)
        smoothed_means, _, smoothed_chols = jax.lax.associative_scan(jax.vmap(sqrt_smoothing_operator),
                                                                     associative_params, reverse=True)
        res = jax.vmap(MVNSqrt)(smoothed_means, smoothed_chols)

    else:
        associative_params = _associative_params(linearization_method, transition_model,
                                                 nominal_trajectory, filter_trajectory, False)
        smoothed_means, _, smoothed_covs = jax.lax.associative_scan(jax.vmap(standard_smoothing_operator),
                                                                    associative_params, reverse=True)
        res = jax.vmap(MVNStandard)(smoothed_means, smoothed_covs)

    return res


def _associative_params(linearization_method, transition_model,
                        nominal_trajectory, filtering_trajectory, sqrt):
    ms, Ps = filtering_trajectory
    nominal_trajectory = jax.tree_map(lambda z: z[:-1], nominal_trajectory)
    if sqrt:
        vmapped_fn = jax.vmap(_sqrt_associative_params, in_axes=[None, None, 0, 0, 0])
    else:
        vmapped_fn = jax.vmap(_standard_associative_params, in_axes=[None, None, 0, 0, 0])
    gs, Es, Ls = vmapped_fn(linearization_method, transition_model, nominal_trajectory, ms[:-1], Ps[:-1])
    g_T, E_T, L_T = ms[-1], jnp.zeros_like(Ps[-1]), Ps[-1]
    return none_or_concat((gs, Es, Ls), (g_T, E_T, L_T), -1)


def _standard_associative_params(linearization_method, transition_model, n_k_1, m, P):
    F, Q, b = linearization_method(transition_model, n_k_1)
    Pp = F @ P @ F.T + Q

    E = jlinalg.solve(Pp, F @ P, sym_pos=True).T

    g = m - E @ (F @ m + b)
    L = P - E @ Pp @ E.T

    return g, E, L


def _sqrt_associative_params(linearization_method, transition_model, n_k_1, m, chol_P):
    F, chol_Q, b = linearization_method(transition_model, n_k_1)
    nx = chol_Q.shape[0]

    Phi = jnp.block([[F @ chol_P, chol_Q],
                     [chol_P, jnp.zeros_like(chol_Q)]])
    Tria_Phi = tria(Phi)
    Phi11 = Tria_Phi[:nx, :nx]
    Phi21 = Tria_Phi[nx:, :nx]
    D = Tria_Phi[nx:, nx:]

    E = jlinalg.solve(Phi11.T, Phi21.T).T
    g = m - E @ (F @ m + b)
    return g, E, D
