import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth.parallel._operators import filtering_operator
from parsmooth._base import MVNParams
from parsmooth._utils import tria


def filter(observations, transition_model, observation_model, linearization_method, sqrt, nominal_trajectory, x0):

    """ Computes the standard and sqrt version of parallel Kalman filter routine given a linearization
         and returns a series of filtered_states TODO:reference
        Parameters
        ----------
        observations: (n, K) array
            array of n observations of dimension K
        transition_model: FunctionalModel or ConditionalMomentsModel

        observation_model: FunctionalModel or ConditionalMomentsModel

        linearization_method: callable
            one of taylor or sigma_points linearization method
        nominal_trajectory: (n, D) array
            points at which to compute the jacobians.
        x0: MVNParams
            prior belief on the initial state distribution
        Returns
        -------
        filtered_states: MVNParams
            list of filtered states
        """

    n_observations = observations.shape[0]

    if sqrt:
        x0 = MVNParams(x0.mean, None, x0.chol)
    else:
        x0 = MVNParams(x0.mean, x0.cov)

    def make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i):
        if sqrt:
            return _sqrt_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i, sqrt)
        return _standard_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i, sqrt)

    @jax.vmap
    def make_params(obs, i):
        return make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, obs, i)

    associative_params = make_params(observations, jnp.arange(n_observations))
    _, filtered_means, filtered_covariances, _, _ = jax.lax.associative_scan(filtering_operator, *associative_params)

    return jax.vmap(MVNParams)(filtered_means, filtered_covariances)


def _standard_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i, sqrt):
    predicate = i == 0

    def _first(_):
        return _standard_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, sqrt)

    def _generic(_):
        return _standard_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y, sqrt)

    return jax.lax.cond(predicate,
                    _first,
                    _generic,
                    None)


def _standard_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, sqrt):

    F, Q, b = linearization_method(transition_model, nominal_trajectory, sqrt)
    H, R, c = linearization_method(observation_model, nominal_trajectory, sqrt)

    m1 = F @ x0.mean + b
    P1 = F @ x0.cov @ F.T + Q

    S = H @ P1 @ H.T + R
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ P1).T
    A = jnp.zeros(F.shape)

    b_std = m1 + K @ (y - H @ m1 - c)
    C = P1 - (K @ S @ K.T)

    temp = (S_invH @ F).T
    eta = temp @ (y - H @ b - c)
    J = temp @ H @ F

    return A, b_std, C, eta, J


def _standard_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y, sqrt):

    F, Q, b = linearization_method(transition_model, nominal_trajectory, sqrt)
    H, R, c = linearization_method(observation_model, nominal_trajectory, sqrt)

    S = H @ Q @ H.T + R
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ Q).T
    A = F - K @ H @ F
    b_std = b + K @ (y - H @ b - c)
    C = Q - K @ H @ Q

    temp = (S_invH @ F).T
    eta = temp @ (y - H @ b - c)
    J = temp @ H @ F

    return A, b_std, C, eta, J


def _sqrt_make_associative_filtering_params(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, i, sqrt):

    predicate = i == 0

    def _first(_):
        return _sqrt_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, sqrt)

    def _generic(_):
        return _sqrt_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y, sqrt)

    return jax.lax.cond(predicate,
                    _first,
                    _generic,
                    None)


def _sqrt_make_associative_filtering_params_first(linearization_method, transition_model, observation_model, nominal_trajectory, x0, y, sqrt):

    F, cholQ, b = linearization_method(transition_model, nominal_trajectory, sqrt)
    H, cholR, c = linearization_method(observation_model, nominal_trajectory, sqrt)

    nx = cholQ.shape[0]
    ny = cholR.shape[0]

    m1 = F @ x0.mean + b
    N1_ = tria(jnp.concatenate((F @ x0.chol, cholQ), axis=1))
    Psi_ = jnp.block([[H @ N1_, cholR], [N1_, jnp.zeros((N1_.shape[0], cholR.shape[1]))]])
    Tria_Psi_ = tria(Psi_)
    Psi11_ = Tria_Psi_[:ny, :ny]
    Psi21_ = Tria_Psi_[ny: ny + nx, :ny]
    Psi22_ = Tria_Psi_[ny: ny + nx, ny:]
    Y1 = Psi11_
    K1 = jlinalg.solve(Psi11_.T, Psi21_.T).T

    A = jnp.zeros(F.shape)
    b_sqr = m1 + K1 @ (y - H @ m1 - c)
    U = Psi22_

    Z1 = jlinalg.solve(Y1, H @ F).T
    eta = jlinalg.solve(Y1.T, Z1.T).T @ (y - H @ b - c)
    if nx > ny:
        Z = jnp.block([Z1, jnp.zeros((nx, nx - ny))])
    else:
        Z = jnp.block([Z1, jnp.zeros((nx, ny - nx))])

    return A, b_sqr, U, eta, Z


def _sqrt_make_associative_filtering_params_generic(linearization_method, transition_model, observation_model, nominal_trajectory, y, sqrt):

    F, cholQ, b = linearization_method(transition_model, nominal_trajectory, sqrt)
    H, cholR, c = linearization_method(observation_model, nominal_trajectory, sqrt)

    nx = cholQ.shape[0]
    ny = cholR.shape[0]
    Psi = jnp.block([[H @ cholQ, cholR], [cholQ, jnp.zeros((nx, ny))]])
    Tria_Psi = tria(Psi)
    Psi11 = Tria_Psi[:ny, :ny]
    Psi21 = Tria_Psi[ny:ny + nx, :ny]
    Psi22 = Tria_Psi[ny:ny + nx, ny:]
    Y = Psi11
    K = jlinalg.solve(Psi11.T, Psi21.T).T
    A = F - K @ H @ F
    b_sqr = b + K @ (y - H @ b - c)
    U = Psi22

    Z1 = jlinalg.solve(Y, H @ F).T
    eta = jlinalg.solve(Y.T, Z1.T).T @ (y - H @ b - c)

    if nx > ny:
        Z = jnp.block([Z1, jnp.zeros((nx, nx - ny))])
    else:
        Z = jnp.block([Z1, jnp.zeros((nx, ny - nx))])

    return A, b_sqr, U, eta, Z

