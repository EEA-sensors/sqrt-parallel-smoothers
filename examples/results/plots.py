import jax
import numpy as np
from matplotlib import pyplot as plt


#### Example 1
with np.load("GPU_RTX3080Ti_population_model_run_time.npz") as loaded:
    ex1_gpu_IEKS_std_seq_time = loaded["gpu_IEKS_std_seq_time"]
    ex1_gpu_IEKS_std_par_time = loaded["gpu_IEKS_std_par_time"]
    ex1_gpu_IPLS_std_seq_time = loaded["gpu_IPLS_std_seq_time"]
    ex1_gpu_IPLS_std_par_time = loaded["gpu_IPLS_std_par_time"]
ex1_lengths_space = np.logspace(2, int(np.log2(1 + 2**7)), num=10, base=2, dtype=int)

plt.figure(figsize=(10,7))
plt.loglog(ex1_lengths_space, ex1_gpu_IEKS_std_seq_time, label="gpu_IEKS_std_seq_GPU", linestyle="-.", linewidth=3)
plt.loglog(ex1_lengths_space, ex1_gpu_IEKS_std_par_time, label="gpu_IEKS_std_par_GPU", linestyle="-.", linewidth=3)
plt.loglog(ex1_lengths_space, ex1_gpu_IPLS_std_seq_time, label="gpu_IPLS_std_seq_GPU", linestyle="-.", linewidth=3)
plt.loglog(ex1_lengths_space, ex1_gpu_IPLS_std_par_time, label="gpu_IPLS_std_par_GPU", linestyle="-.", linewidth=3)
plt.grid(True, which="both")
plt.title("GPU runtime - Population model")
plt.legend()
plt.show()

#### Example 2
with np.load("GPU_RTX3080Ti_bearing_only_run_time.npz") as loaded:
   ex2_gpu_cubature_sqrt_par_mean_time = loaded["gpu_cubature_sqrt_par_mean_time"]
   ex2_gpu_cubature_sqrt_seq_mean_time = loaded["gpu_cubature_sqrt_seq_mean_time"]
   ex2_gpu_extended_sqrt_par_mean_time = loaded["gpu_extended_sqrt_par_mean_time"]
   ex2_gpu_extended_sqrt_seq_mean_time = loaded["gpu_extended_sqrt_seq_mean_time"]
ex2_lengths_space = np.logspace(2, int(np.log2(5000)), num=10, base=2, dtype=int)

plt.figure(figsize=(10,7))
plt.loglog(ex2_lengths_space, ex2_gpu_cubature_sqrt_par_mean_time, label="gpu_cubature_sqrt_par_mean_GPU", linestyle="-.", linewidth=3)
plt.loglog(ex2_lengths_space, ex2_gpu_cubature_sqrt_seq_mean_time, label="gpu_cubature_sqrt_seq_mean_GPU", linestyle="-.", linewidth=3)
plt.loglog(ex2_lengths_space, ex2_gpu_extended_sqrt_par_mean_time, label="gpu_extended_sqrt_par_mean_GPU", linestyle="-.", linewidth=3)
plt.loglog(ex2_lengths_space, ex2_gpu_extended_sqrt_seq_mean_time, label="gpu_extended_sqrt_seq_mean_GPU", linestyle="-.", linewidth=3)
plt.grid(True, which="both")
plt.title("GPU runtime - Bearing only")
plt.legend()
plt.show()



#### Example 3
Ts = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
with np.load("ell_float32_extended_std_runtime15.npz") as loaded:
    rts_fl32_IEKS_std_par_ell = loaded["rts_gpu_IEKS_std_par_ell"]

with np.load("ell_float32_extended_sqrt_runtime15.npz") as loaded:
    rts_fl32_IEKS_sqrt_par_ell = loaded["rts_gpu_IEKS_sqrt_par_ell"]

rts_fl32_gpu_extended_sqrt_par_ell = np.mean(np.isnan(rts_fl32_IEKS_sqrt_par_ell),axis=1)*100
rts_fl32_gpu_extended_std_par_ell = np.mean(np.isnan(rts_fl32_IEKS_std_par_ell),axis=1)*100

plt.plot(Ts, rts_fl32_gpu_extended_sqrt_par_ell, '--*', label="sqrt-extended")
plt.plot(Ts, rts_fl32_gpu_extended_std_par_ell, '--*', label="std-extended")
plt.legend()
plt.show()

with np.load("ell_float32_cubature_std_runtime15.npz") as loaded:
    rts_fl32_ICKS_std_par_ell = loaded["gpu_ICKS_std_par_ell"]

with np.load("ell_float32_cubature_sqrt_runtime15.npz") as loaded:
    rts_fl32_ICKS_sqrt_par_ell = loaded["gpu_ICKS_sqrt_par_ell"]

rts_fl32_gpu_cubature_sqrt_par_ell = np.mean(np.isnan(rts_fl32_ICKS_sqrt_par_ell),axis=1)*100
rts_fl32_gpu_cubature_std_par_ell = np.mean(np.isnan(rts_fl32_ICKS_std_par_ell),axis=1)*100

plt.plot(Ts, rts_fl32_gpu_cubature_sqrt_par_ell, '--*', label="sqrt-cubature")
plt.plot(Ts, rts_fl32_gpu_cubature_std_par_ell, '--*', label="std-cubature")
plt.legend()
plt.show()

