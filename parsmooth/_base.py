import itertools
from typing import NamedTuple, Callable, Any


class MVNStandard(NamedTuple):
    mean: Any
    cov: Any


class MVNSqrt(NamedTuple):
    mean: Any
    chol: Any


class FunctionalModel(NamedTuple):
    function: Callable
    mvn: MVNStandard or MVNSqrt


class ConditionalMomentsModel(NamedTuple):
    conditional_mean: Callable
    conditional_variance: Callable


def are_inputs_compatible(*y):
    a, b = itertools.tee(map(type, y))
    _ = next(b, None)
    ok = sum(map(lambda u: u[0] == u[1], zip(a, b)))
    if not ok:
        raise TypeError(f"All inputs should have the same type. {y} was given")
