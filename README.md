# sqrt-parallel-smoothers
Companion code for the papers xxx [1] and xxx [2]. 

## Quick description

A generic library written in JAX for parallel-in-time linear and non-linear Gaussian (iterated) smoothing. 
This code supports the covariance-based parallel formulation predict update algorithm [1], 
as well as the more robust square-root versions [2]. Several linearization algorithms (extended and cubature)
can be used, both in a sequential and parallel fashion. 

Moreover, when using the iterated smoother method, the gradients of the optimal nominal trajectory, and therefore of the resulting (pseudo) log-likelihood, are computed efficiently 
using the implicit function theorem corresponding to the fixed point equation. This provides more stable solutions as well as
constant memory computation rules.

For more details, we refer to our article xxx [2].

## Installation

This package has different requirements depending on your intended use for it. 

### Minimal installation
If you simply want to use it as part of your own code, then we recommend the following steps
1. Create and activate a virtual environment using your favourite method (`conda create ...` or `python -m venv path` 
   for example).
2. Install your required version of JAX:
   * GPU (preferred): at the time of writing JAX **only supports the GPU backend for linux distributions**. 
     You will need to make sure you have the proper CUDA (at the time of writing 11.4) version installed
     and then run (at the time of writing)
     ```bash
     pip install --upgrade pip
     pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
     ```
   * CPU (no support for parallelisation): at the time of writing this is supported for **linux and MacOS** users only. 
     Run (at the time of writing) 
    ```bash
     pip install --upgrade pip
     pip install --upgrade "jax[cpu]"
     ```
   * For more up-to-date installation instructions we refer to JAX github page https://github.com/google/jax.
3. Run `pip install -r requirements.txt`
4. Run `python setup.py [develop|install]` depending on if you plan to work on the source code or not.

### Additional test requirements
If you plan on running the tests, please run `pip install -r requirements-test.txt`

## Contact information
This library was developed by [Adrien Corenflos](@adriencorenflos) and [Fatemeh Yaghoobi](@Fatemeh-Yaghoobi). For any code related question feel free to open a discussion
in the issues tab, and for more technical questions please send an email to the article corresponding email address 
fatemeh[dot]yaghoobi[at]aalto[dot]fi.

## References
[1]:
[2]: 

## How to cite
If you use our code/build on our work, please cite us! The correct bibtex entries are included in the repository.


