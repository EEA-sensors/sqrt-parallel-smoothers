import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing
from bearing_data import get_data, make_parameters

s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.5  # Observation noise (stddev)
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
dt = 0.01  # discretization time step
qc = 0.01  # discretization noise0

qw = 0.1  # discretization noise

Q, R, observation_function, transition_function = make_parameters(qc, qw, r, dt, s1, s2)

chol_Q = jnp.linalg.cholesky(Q)
chol_R = jnp.linalg.cholesky(R)

m0 = jnp.array([-4., -1., 2., 7., 3.])
chol_P0 = jnp.eye(5)
P0 = jnp.eye(5)

init = MVNStandard(m0, P0)
chol_init = MVNSqrt(m0, chol_P0)


sqrt_transition_model = FunctionalModel(transition_function, MVNSqrt(jnp.zeros((5,)), chol_Q))
transition_model = FunctionalModel(transition_function, MVNStandard(jnp.zeros((5,)), Q))

sqrt_observation_model = FunctionalModel(observation_function, MVNSqrt(jnp.zeros((2,)), chol_R))
observation_model = FunctionalModel(observation_function, MVNStandard(jnp.zeros((2,)), R))

################################################### Data generation #################################
n_run = 100
Ts = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
# for t, T in enumerate(Ts):
#     data = []
#     for i in range(n_run):
#         _, _, ys = get_data(x0, dt, r, T, s1, s2)
#         jnp.savez("Data_robustness_20observations_100runs/data-T" + str(t+1) + "-Run" + str(i+1), data=ys)
######################################################################################################
# select the observation data
k = 16
one_T = [Ts[k]]

def func(method, Ts, runtime=n_run, n_iter=20, sqrt=True):
    ell_par = []
    for i, T in enumerate(Ts):
        print(f"Length {i + 1} out of {len(Ts)}")
        ell_par_res = []
        for j in range(runtime):
            with np.load("Data_robustness_20observations_100runs/data-T" + str(i + 1 + k) + '-Run' + str(j + 1) + ".npz") as data:
                ys = data["data"]

            if sqrt:
                initial_states_sqrt = MVNSqrt(jnp.repeat(jnp.array([[-1., -1., 6., 4., 2.]]), T + 1, axis=0),
                                              jnp.repeat(jnp.eye(5).reshape(1, 5, 5), T + 1, axis=0))
                args = ys, initial_states_sqrt, n_iter

            else:
                initial_states = MVNStandard(jnp.repeat(jnp.array([[-1., -1., 6., 4., 2.]]), T + 1, axis=0),
                                             jnp.repeat(jnp.eye(5).reshape(1, 5, 5), T + 1, axis=0))
                args = ys, initial_states, n_iter

            _, ell = method(*args)

            ell_par_res.append(ell)
            print(f"run {j + 1} out of {runtime}", end="\r")

        ell_par.append(ell_par_res)
    print()

    return ell_par

# Extended
def IEKS_std_par(observations, initial_points, iteration):
    std_par_res, ell = iterated_smoothing(observations, init, transition_model, observation_model,
                                          extended, initial_points, True,
                                          criterion=lambda i, *_: i < iteration,
                                          return_loglikelihood=True)
    return std_par_res, ell


def IEKS_sqrt_par(observations, initial_points_sqrt, iteration):
    sqrt_par_res, ell = iterated_smoothing(observations, chol_init, sqrt_transition_model, sqrt_observation_model,
                                           extended, initial_points_sqrt, True,
                                           criterion=lambda i, *_: i < iteration,
                                           return_loglikelihood=True)
    return sqrt_par_res, ell


gpu_IEKS_std_par = jit(IEKS_std_par, backend="gpu")
gpu_IEKS_sqrt_par = jit(IEKS_sqrt_par, backend="gpu")

gpu_IEKS_std_par_ell = func(gpu_IEKS_std_par, one_T, sqrt=False)
divergence_rate_extended_std_par_ell = np.mean(np.isnan(gpu_IEKS_std_par_ell), axis=1)*100


gpu_IEKS_sqrt_par_ell = func(gpu_IEKS_sqrt_par, one_T, sqrt=True)
divergence_rate_extended_sqrt_par_ell = np.mean(np.isnan(gpu_IEKS_sqrt_par_ell), axis=1)*100

print("T = ", one_T)
print("Standard\n")
print(divergence_rate_extended_std_par_ell)
print("Sqrt\n")
print(divergence_rate_extended_sqrt_par_ell)

# for k = 16, i.e., T = 6500 I get the following results
# Standard divergence rate: 4.
# Sqrt divergence rate: 5.