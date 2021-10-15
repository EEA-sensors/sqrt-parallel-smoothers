from typing import Callable, Optional, Union

from jax import numpy as jnp

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, ConditionalMomentsModel
from parsmooth._pathwise_sampler import _par_sampling, _seq_sampling
from parsmooth._utils import fixed_point
from parsmooth.parallel._filtering import filtering as par_filtering
from parsmooth.parallel._smoothing import smoothing as par_smoothing
from parsmooth.sequential._filtering import filtering as seq_filtering
from parsmooth.sequential._smoothing import smoothing as seq_smoothing


def filtering(observations: jnp.ndarray,
              x0: Union[MVNSqrt, MVNStandard],
              transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              observation_model: Union[FunctionalModel, ConditionalMomentsModel],
              linearization_method: Callable,
              nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
              parallel: bool = True,
              return_loglikelihood: bool = False):
    if parallel:
        return par_filtering(observations, x0, transition_model, observation_model, linearization_method,
                             nominal_trajectory, return_loglikelihood)
    return seq_filtering(observations, x0, transition_model, observation_model, linearization_method,
                         nominal_trajectory, return_loglikelihood)


def smoothing(transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              filter_trajectory: Union[MVNSqrt, MVNStandard],
              linearization_method: Callable, nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
              parallel: bool = True):
    if parallel:
        return par_smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory)
    return seq_smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory)


def filter_smoother(observations: jnp.ndarray,
                    x0: Union[MVNSqrt, MVNStandard],
                    transition_model: Union[FunctionalModel, ConditionalMomentsModel],
                    observation_model: Union[FunctionalModel, ConditionalMomentsModel],
                    linearization_method: Callable,
                    nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
                    parallel: bool = True):
    filter_trajectory = filtering(observations, x0, transition_model, observation_model, linearization_method,
                                  nominal_trajectory, parallel)
    return smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory, parallel)


def _default_criterion(_i, nominal_traj_prev, curr_nominal_traj):
    return jnp.mean((nominal_traj_prev.mean - curr_nominal_traj.mean) ** 2) > 1e-6


def iterated_smoothing(observations: jnp.ndarray,
                       x0: Union[MVNSqrt, MVNStandard],
                       transition_model: Union[FunctionalModel, ConditionalMomentsModel],
                       observation_model: Union[FunctionalModel, ConditionalMomentsModel],
                       linearization_method: Callable,
                       init_nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
                       parallel: bool = True,
                       criterion: Callable = _default_criterion,
                       return_loglikelihood: bool = False):
    if init_nominal_trajectory is None:
        init_nominal_trajectory = filter_smoother(observations, x0, transition_model, observation_model,
                                                  linearization_method, None, parallel)

    def fun_to_iter(curr_nominal_traj):
        return filter_smoother(observations, x0, transition_model, observation_model, linearization_method,
                               curr_nominal_traj, parallel)

    nominal_traj = fixed_point(fun_to_iter, init_nominal_trajectory, criterion)
    if return_loglikelihood:
        _, ell = filtering(observations, x0, transition_model, observation_model, linearization_method,
                           nominal_traj, parallel, return_loglikelihood=True)
        return nominal_traj, ell
    return nominal_traj


def sampling(key: jnp.ndarray,
             n_samples: int,
             transition_model: Union[FunctionalModel, ConditionalMomentsModel],
             filter_trajectory: MVNSqrt or MVNStandard,
             linearization_method: Callable,
             nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
             parallel: bool = True):
    nominal_trajectory = nominal_trajectory or smoothing(transition_model, filter_trajectory, linearization_method,
                                                         None, parallel)
    if parallel:
        return _par_sampling(key, n_samples, transition_model, filter_trajectory, linearization_method,
                             nominal_trajectory)
    return _seq_sampling(key, n_samples, transition_model, filter_trajectory, linearization_method, nominal_trajectory)
