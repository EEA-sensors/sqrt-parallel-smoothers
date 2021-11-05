import numpy as np


with np.load("iterated_res_mean.npz") as loaded:
    mean, expected_mean, cov, expected_cov = loaded["mean"], loaded["expected_mean"], loaded["cov"], loaded["expected_cov"]

with np.load("iterated_res_mean_new.npz") as loaded:
    mean1, expected_mean1, cov1, expected_cov1 = loaded["mean"], loaded["expected_mean"], loaded["cov"], loaded["expected_cov"]
n = 267
print()
print("Fixed point method")
print("mean = \n ", mean[n])
print("expected_mean = \n ", expected_mean[n])
# print("cov = \n", cov[n])
# print("expected_cov = \n", expected_cov[n])
print("Without fixed point method")
print("mean1 = \n ", mean1[n])
print("expected_mean1 = \n ", expected_mean1[n])
# print("cov1 = \n", cov1[n])
# print("expected_cov1 = \n", expected_cov1[n])