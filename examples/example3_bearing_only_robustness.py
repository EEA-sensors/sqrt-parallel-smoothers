import jax
import pickle
from jax import jit
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.methods import iterated_smoothing
from bearing_data import get_data, make_parameters

s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.5  # Observation noise (stddev)
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
dt = 0.01  # discretization time step
qc = 0.01  # discretization noise
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

n_run = 15
Ts = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]

# Data_robustness = []
# for t, T in enumerate(Ts):
#     data = []
#     for i in range(n_run):
#         _, true_states, ys = get_data(x0, dt, r, T, s1, s2)
#         data.append({'ys': ys})
#     Data_robustness.append(data)
#
# with  open("new_outputs/data_robustness.pkl", "wb") as open_file:
#     pickle.dump(Data_robustness, open_file)

with open("results/data_robustness.pkl", "rb") as open_file:
    Data_robustness = pickle.load(open_file)


# Parallel

def func(method, Ts, data, runtime=n_run, n_iter=20, sqrt=True, mth="extended_std"):
    ell_par = []
    for i, T in enumerate(Ts):
        print(f"Length {i + 1} out of {len(Ts)}")
        ell_par_res = []
        for j in range(runtime):
            ys = data[i][j]['ys']
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
        # jnp.savez("new_outputs/ell_float32" + mth + "-" + str(i), ell_fl32=jnp.array(ell_par_res))
    print()

    return ell_par

## Extended
def IEKS_std_par(observations, initial_points, iteration):
    std_par_res, ell = iterated_smoothing(observations, init, transition_model, observation_model,
                                          extended, initial_points, True,
                                          criterion=lambda i, *_: i < iteration,
                                          return_loglikelihood = True)
    return std_par_res, ell


def IEKS_sqrt_par(observations, initial_points_sqrt, iteration):
    sqrt_par_res, ell = iterated_smoothing(observations, chol_init, sqrt_transition_model, sqrt_observation_model,
                                           extended, initial_points_sqrt, True,
                                           criterion=lambda i, *_: i < iteration,
                                           return_loglikelihood = True)
    return sqrt_par_res, ell

gpu_IEKS_std_par = jit(IEKS_std_par, backend="gpu")
gpu_IEKS_sqrt_par = jit(IEKS_sqrt_par, backend="gpu")

gpu_IEKS_std_par_ell = func(gpu_IEKS_std_par, Ts, Data_robustness, sqrt=False)
# jnp.savez("ell_float32_extended_std_runtime15",
#           rts_gpu_IEKS_std_par_ell=gpu_IEKS_std_par_ell)

gpu_IEKS_sqrt_par_ell = func(gpu_IEKS_sqrt_par, Ts, Data_robustness, sqrt=True, mth="extended_sqrt")
# jnp.savez("ell_float32_extended_sqrt_runtime15",
#           rts_gpu_IEKS_sqrt_par_ell=gpu_IEKS_sqrt_par_ell)

# Ts = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
# mth='extended_sqrt'
# rts_gpu_IEKS_sqrt_par_ell = []
# for t in range(len(Ts)):
#     with np.load("new_outputs/ell_float32"+mth+"-"+str(t)+".npz") as loaded:
#         rts_gpu_IEKS_sqrt_par_ell.append(loaded["ell_fl32"])
#
# np.savez("new_outputs/ell_float32_extended_sqrt_runtime15",
#           rts_gpu_IEKS_sqrt_par_ell = rts_gpu_IEKS_sqrt_par_ell)

## Cubature

def ICKS_std_par(observations, initial_points, iteration):
    std_par_res, ell = iterated_smoothing(observations, init, transition_model, observation_model,
                                                 cubature, initial_points, True,
                                                 criterion=lambda i, *_: i < iteration,
                                                 return_loglikelihood = True)
    return std_par_res, ell


def ICKS_sqrt_par(observations, initial_points_sqrt, iteration):
    sqrt_par_res, ell = iterated_smoothing(observations, chol_init, sqrt_transition_model, sqrt_observation_model,
                                                      cubature, initial_points_sqrt, True,
                                                      criterion=lambda i, *_: i < iteration,
                                                      return_loglikelihood = True)
    return sqrt_par_res, ell

gpu_ICKS_std_par = jit(ICKS_std_par, backend="gpu")
gpu_ICKS_sqrt_par = jit(ICKS_sqrt_par, backend="gpu")

gpu_ICKS_std_par_ell = func(gpu_ICKS_std_par, Ts, Data_robustness, sqrt=False, mth="cubature_std")
# jnp.savez("ell_float32_cubature_std_runtime15",
#           gpu_ICKS_std_par_ell=gpu_ICKS_std_par_ell)

gpu_ICKS_sqrt_par_ell = func(gpu_ICKS_sqrt_par, Ts, Data_robustness, sqrt=True, mth="cubature_sqrt")
# jnp.savez("ell_float32_cubature_sqrt_runtime15",
#           gpu_ICKS_sqrt_par_ell=gpu_ICKS_sqrt_par_ell)

