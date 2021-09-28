import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._utils import tria


def standard_filtering_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        a_i, b_i, C_i, eta_i, J_i
    elem2: tuple of array
        a_j, b_j, C_j, eta_j, J_j

    Returns
    -------
    elem12: tuple of array
        ...
    """
    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    dim = b1.shape[0]

    I_dim = jnp.eye(dim)

    IpCJ = I_dim + jnp.dot(C1, J2)
    IpJC = I_dim + jnp.dot(J2, C1)

    AIpCJ_inv = jlinalg.solve(IpCJ.T, A2.T, sym_pos=False).T
    AIpJC_inv = jlinalg.solve(IpJC.T, A1, sym_pos=False).T

    A = jnp.dot(AIpCJ_inv, A1)
    b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
    C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.T)) + C2
    eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
    J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
    return A, b, C, eta, J


def sqrt_filtering_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        ...
    elem2: tuple of array
        ...

    Returns
    -------

    """
    A1, b1, U1, eta1, Z1 = elem1
    A2, b2, U2, eta2, Z2 = elem2

    nx = Z2.shape[0]

    Xi = jnp.block([[U1.T @ Z2, jnp.eye(nx)],
                    [Z2, jnp.zeros_like(A1)]])
    tria_xi = tria(Xi)
    Xi11 = tria_xi[:nx, :nx]
    Xi21 = tria_xi[nx: nx + nx, :nx]
    Xi22 = tria_xi[nx: nx + nx, nx:]

    A = A2 @ A1 - jlinalg.solve_triangular(Xi11, U1.T @ A2.T, lower=True).T @ Xi21.T @ A1
    b = A2 @ (jnp.eye(nx) - jlinalg.solve_triangular(Xi11, U1.T, lower=True).T @ Xi21.T) @ (b1 + U1 @ U1.T @ eta2) + b2
    U = tria(jnp.concatenate([jlinalg.solve_triangular(Xi11, U1.T @ A2.T, lower=True).T, U2], axis=1))
    eta = A1.T @ (jnp.eye(nx) - jlinalg.solve_triangular(Xi11, Xi21.T, lower=True, trans=True).T @ U1.T) @ (
            eta2 - Z2 @ Z2.T @ b1) + eta1
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    return A, b, U, eta, Z


def standard_smoothing_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        ...
    elem2: tuple of array
        ...

    Returns
    -------

    """
    g1, E1, L1 = elem1
    g2, E2, L2 = elem2

    g = E2 @ g1 + g2
    E = E2 @ E1
    L = E2 @ L1 @ E2.T + L2
    return g, E, L


def sqrt_smoothing_operator(elem1, elem2):
    """

    Parameters
    ----------
    elem1: tuple of array
        g_i, E_i, D_i
    elem2: tuple of array
        g_j, E_j, D_j

    Returns
    -------

    """
    g1, E1, D1 = elem1
    g2, E2, D2 = elem2

    g = E2 @ g1 + g2
    E = E2 @ E1
    D = tria(jnp.concatenate([E2 @ D1, D2], axis=1))

    return g, E, D
