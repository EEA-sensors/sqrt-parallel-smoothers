import jax
import jax.numpy as jnp
from jax import jit
import time
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from parsmooth._base import MVNStandard, ConditionalMomentsModel
from parsmooth.linearization import cubature, extended
from parsmooth.methods import iterated_smoothing


from population_model import make_parameters
from population_model import get_data as population_data

seed = 2
key =jax.random.PRNGKey(seed)

T = 1 + 2**7
lam = 10.
Q = jnp.array([[0.3**2]])
m0 = jnp.array([jnp.log(7.)])

true_states, observations = population_data(m0, T, Q, lam, key)

dim_x = 1
mean_f, cov_f, mean_h, cov_h, chol_f, chol_h = make_parameters(lam, Q)
transition_model = ConditionalMomentsModel(mean_f, cov_f)
observation_model = ConditionalMomentsModel(mean_h, cov_h)

m0 = np.random.randn(dim_x)
chol_x0 = np.random.randn(dim_x, dim_x)
cov_x0 = chol_x0 @ chol_x0.T
x0 = MVNStandard(m0, cov_x0)

m_nominal = np.random.randn(T + 1, dim_x)
P_nominal = jnp.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
x_nominal = MVNStandard(m_nominal, P_nominal)

iteration = 100
# extended
extended_std_seq = iterated_smoothing(observations, x0, transition_model, observation_model,
                                      extended, x_nominal, False,
                                      criterion=lambda i, *_: i < iteration)

extended_std_par = iterated_smoothing(observations, x0, transition_model, observation_model,
                                      extended, x_nominal, True,
                                      criterion=lambda i, *_: i < iteration)

np.testing.assert_array_almost_equal(extended_std_seq.mean,
                                     extended_std_par.mean, decimal=9)

# cubature
cubature_std_seq = iterated_smoothing(observations, x0, transition_model, observation_model,
                                      cubature, x_nominal, False,
                                      criterion=lambda i, *_: i < iteration)
cubature_std_par = iterated_smoothing(observations, x0, transition_model, observation_model,
                                      cubature, x_nominal, True,
                                      criterion=lambda i, *_: i < iteration)
np.testing.assert_array_almost_equal(cubature_std_seq.mean,
                                     cubature_std_par.mean, decimal=9)


def func(method, lengths, n_runs=100, n_iter=100):
    res_mean = []
    for k, j in enumerate(lengths):
        print(f"Iteration {k + 1} out of {len(lengths)}", end="\r")
        observations_slice = observations[:j]

        init_linearizations_points_slice = x_nominal.mean[:j + 1]
        init_linearizations_covs_slice = x_nominal.cov[:j + 1]
        init_linearizations_states = MVNStandard(init_linearizations_points_slice, init_linearizations_covs_slice)
        args = observations_slice, init_linearizations_states, n_iter

        s = method(*args)
        s.mean.block_until_ready()
        run_times = []
        for _ in range(n_runs):
            tic = time.time()
            s_states = method(*args)
            s_states.mean.block_until_ready()
            toc = time.time()
            run_times.append(toc - tic)
        res_mean.append(np.mean(run_times))
    print()
    return np.array(res_mean)

lengths_space = np.logspace(2, int(np.log2(T)), num=10, base=2, dtype=int)


def IEKS_std_seq(y, x_nominal_trajectory, iteration):
    return iterated_smoothing(y, x0, transition_model, observation_model,
                              extended, x_nominal_trajectory, False,
                              criterion=lambda i, *_: i < iteration)


def IEKS_std_par(y, x_nominal_trajectory, iteration):
    return iterated_smoothing(y, x0, transition_model, observation_model,
                              extended, x_nominal_trajectory, True,
                              criterion=lambda i, *_: i < iteration)


def IPLS_std_seq(y, x_nominal_trajectory, iteration):
    return iterated_smoothing(y, x0, transition_model, observation_model,
                              cubature, x_nominal_trajectory, False,
                              criterion=lambda i, *_: i < iteration)


def IPLS_std_par(y, x_nominal_trajectory, iteration):
    return iterated_smoothing(y, x0, transition_model, observation_model,
                              cubature, x_nominal_trajectory, True,
                              criterion=lambda i, *_: i < iteration)

gpu_IEKS_std_seq = jit(IEKS_std_seq, backend="gpu")
gpu_IEKS_std_par = jit(IEKS_std_par, backend="gpu")
gpu_IPLS_std_seq = jit(IPLS_std_seq, backend="gpu")
gpu_IPLS_std_par = jit(IPLS_std_par, backend="gpu")

gpu_IEKS_std_seq_time = func(gpu_IEKS_std_seq, lengths_space)
gpu_IEKS_std_par_time = func(gpu_IEKS_std_par, lengths_space)
gpu_IPLS_std_seq_time = func(gpu_IPLS_std_seq, lengths_space)
gpu_IPLS_std_par_time = func(gpu_IPLS_std_par, lengths_space)

plt.figure(figsize=(10, 7))
plt.loglog(lengths_space, gpu_IEKS_std_seq_time, label="gpu_IEKS_std_seq_GPU", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_IEKS_std_par_time, label="gpu_IEKS_std_par_GPU", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_IPLS_std_seq_time, label="gpu_IPLS_std_seq_GPU", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_IPLS_std_par_time, label="gpu_IPLS_std_par_GPU", linestyle="-.", linewidth=3)
plt.grid(True, which="both")
plt.title("GPU runtime using rtx 3080 Ti")
plt.legend()
plt.show()
############################################################################################################
# jnp.savez("GPU_RTX3080Ti_population_model_run_time.npz",
#           gpu_IEKS_std_seq_time=gpu_IEKS_std_seq_time,
#           gpu_IEKS_std_par_time=gpu_IEKS_std_par_time,
#           gpu_IPLS_std_seq_time=gpu_IPLS_std_seq_time,
#           gpu_IPLS_std_par_time=gpu_IPLS_std_par_time)
