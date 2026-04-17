# Parallel square-root statistical linear regression for inference in nonlinear state space models

A generic library for linear and non-linear Gaussian smoothing problems.
The code leverages JAX and implements several linearization algorithms,
both in a sequential and parallel fashion, as well as low-memory cost algorithms computing gradients of required
quantities
(such as the pseudo-loglikelihood of the system).

This code was written by [Adrien Corenflos](https://github.com/AdrienCorenflos)
and [Fatemeh Yaghoobi](https://github.com/Fatemeh-Yaghoobi) as a companion code for the article
"Parallel square-root statistical linear regression for inference in nonlinear state space models"
by Fatemeh Yaghoobi, Adrien Corenflos, Sakira Hassan, and Simo Särkkä, ArXiv link: [https://arxiv.org/abs/2207.00426](https://arxiv.org/abs/2207.00426), Journal link [https://epubs.siam.org/doi/10.1137/23M156121X](https://epubs.siam.org/doi/10.1137/23M156121X)

## Installation

1. Create a virtual environment and clone this repository
2. Install JAX (preferably with GPU support) following https://github.com/google/jax#installation
3. Run `pip install .`
4. (optional) If you want to run the examples, run `pip install -r requirements-examples.txt`

## Examples

Example uses (reproducing the experiments of our paper) can be found in the [examples folder](../main/notebooks). More
low-level examples can be found in the
[test folder](../main/tests).

## How to cite

If you find this work useful, please cite us in the following way:

```
@misc{yaghoobi2022sqrt,
author = {Yaghoobi, Fatemeh and Corenflos, Adrien and Hassan, Sakira and S\"{a}rkk\"{a}, Simo},
title = {Parallel Square-Root Statistical Linear Regression for Inference in Nonlinear State Space Models},
journal = {SIAM Journal on Scientific Computing},
volume = {47},
number = {2},
pages = {B454-B476},
year = {2025},
doi = {10.1137/23M156121X},
}
```
