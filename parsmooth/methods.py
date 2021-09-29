from typing import Callable, Optional

from jax import numpy as jnp

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth._pathwise_sampler import _par_sampling, _seq_sampling
from parsmooth._utils import fixed_point
from parsmooth.parallel._filtering import filtering as par_filtering
from parsmooth.parallel._smoothing import smoothing as par_smoothing
from parsmooth.sequential._filtering import filtering as seq_filtering
from parsmooth.sequential._smoothing import smoothing as seq_smoothing


def filtering(observations: jnp.ndarray,
              x0: MVNStandard or MVNSqrt,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method: Callable,
              nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None,
              parallel: bool = True):
    if parallel:
        return par_filtering(observations, x0, transition_model, observation_model, linearization_method,
                             nominal_trajectory)
    return seq_filtering(observations, x0, transition_model, observation_model, linearization_method,
                         nominal_trajectory)


def smoothing(transition_model: FunctionalModel, filter_trajectory: MVNSqrt or MVNStandard,
              linearization_method: Callable, nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None,
              parallel: bool = True):
    if parallel:
        return par_smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory)
    return seq_smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory)


def _filter_smoother(observations: jnp.ndarray,
                     x0: MVNStandard or MVNSqrt,
                     transition_model: FunctionalModel,
                     observation_model: FunctionalModel,
                     linearization_method: Callable,
                     nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None,
                     parallel: bool = True):
    filter_trajectory = filtering(observations, x0, transition_model, observation_model, linearization_method,
                                  nominal_trajectory, parallel)
    return smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory, parallel)


def _default_criterion(_i, nominal_traj_prev, curr_nominal_traj):
    return jnp.mean((nominal_traj_prev.mean - curr_nominal_traj.mean) ** 2) > 1e-6


def iterated_smoothing(observations: jnp.ndarray,
                       x0: MVNStandard or MVNSqrt,
                       transition_model: FunctionalModel,
                       observation_model: FunctionalModel,
                       linearization_method: Callable,
                       init_nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None,
                       parallel: bool = True,
                       criterion: Callable = _default_criterion):
    if init_nominal_trajectory is None:
        init_nominal_trajectory = _filter_smoother(observations, x0, transition_model, observation_model,
                                                   linearization_method, None, parallel)

    def fun_to_iter(curr_nominal_traj):
        return _filter_smoother(observations, x0, transition_model, observation_model, linearization_method,
                                curr_nominal_traj, parallel)

    return fixed_point(fun_to_iter, init_nominal_trajectory, criterion)


def sampling(key: jnp.ndarray,
             n_samples: int,
             transition_model: FunctionalModel,
             filter_trajectory: MVNSqrt or MVNStandard,
             linearization_method: Callable,
             nominal_trajectory: Optional[MVNStandard or MVNSqrt] = None,
             parallel: bool = True):
    nominal_trajectory = nominal_trajectory or smoothing(transition_model, filter_trajectory, linearization_method,
                                                         None, parallel)
    if parallel:
        return _par_sampling(key, n_samples, transition_model, filter_trajectory, linearization_method,
                             nominal_trajectory)
    return _seq_sampling(key, n_samples, transition_model, filter_trajectory, linearization_method, nominal_trajectory)
