"""Utility functions for LISA response computations."""

from typing import Tuple

import jax
import jax.numpy as jnp

# Physical constants
C_inv = 3.3356409519815204e-09  # Inverse speed of light in s/m
YRSID_SI = 31558149.763545603  # Seconds in a sidereal year


@jax.jit
def get_basis_vecs(
    lam: float, beta: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the basis vectors for gravitational wave polarization.

    Parameters
    ----------
    lam : float
        Ecliptic longitude in radians
    beta : float
        Ecliptic latitude in radians

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Tuple of (u, v, k) basis vectors, each of shape (3,)
        where u and v are polarization vectors and k is propagation direction
    """
    cosbeta = jnp.cos(beta)
    sinbeta = jnp.sin(beta)
    coslam = jnp.cos(lam)
    sinlam = jnp.sin(lam)

    # Note: These basis vectors match the CUDA implementation
    v = jnp.array([-sinbeta * coslam, -sinbeta * sinlam, cosbeta])
    u = jnp.array([sinlam, -coslam, 0.0])
    k = jnp.array([-cosbeta * coslam, -cosbeta * sinlam, -sinbeta])

    return u, v, k


@jax.jit
def dot_product_1d(arr1: jnp.ndarray, arr2: jnp.ndarray) -> float:
    """Compute dot product of two 3D vectors.

    Parameters
    ----------
    arr1 : jnp.ndarray
        First vector of shape (3,)
    arr2 : jnp.ndarray
        Second vector of shape (3,)

    Returns
    -------
    float
        Scalar dot product
    """
    return jnp.sum(arr1 * arr2)


@jax.jit
def xi_projections(
    u: jnp.ndarray, v: jnp.ndarray, n: jnp.ndarray
) -> Tuple[float, float]:
    """Compute xi projections for gravitational wave response.

    Parameters
    ----------
    u : jnp.ndarray
        Polarization vector u of shape (3,)
    v : jnp.ndarray
        Polarization vector v of shape (3,)
    n : jnp.ndarray
        Unit vector along spacecraft arm of shape (3,)

    Returns
    -------
    Tuple[float, float]
        Tuple of (xi_p, xi_c) projections for plus and cross polarizations
    """
    u_dot_n = dot_product_1d(u, n)
    v_dot_n = dot_product_1d(v, n)

    xi_p = 0.5 * ((u_dot_n * u_dot_n) - (v_dot_n * v_dot_n))
    xi_c = u_dot_n * v_dot_n

    return xi_p, xi_c


@jax.jit
def normalize_vector(vec: jnp.ndarray) -> jnp.ndarray:
    """Normalize a 3D vector.

    Parameters
    ----------
    vec : jnp.ndarray
        Vector of shape (3,)

    Returns
    -------
    jnp.ndarray
        Normalized vector of shape (3,)
    """
    norm = jnp.sqrt(jnp.sum(vec * vec))
    return vec / norm


def get_factorial(n: int) -> float:
    """Compute factorial of n.

    Args:
        n: Integer to compute factorial of

    Returns:
        Factorial value as float
    """
    if n <= 1:
        return 1.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


def generate_factorial_array(max_n: int) -> jnp.ndarray:
    """Generate array of factorials up to max_n.

    Parameters
    ----------
    max_n : int
        Maximum value to compute factorial for

    Returns
    -------
    jnp.ndarray
        Array of factorials [0!, 1!, 2!, ..., max_n!]
    """
    factorials = jnp.array([get_factorial(i) for i in range(max_n + 1)])
    return factorials


@jax.jit
def xyz_to_aet(
    X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transform TDI channels from XYZ to AET.

    This function implements the standard AET transformation used in LISA data analysis,
    which provides better noise characteristics and scientific interpretation.

    The transformation equations are:
    - A = (Z - X) / √2
    - E = (X - 2Y + Z) / √6
    - T = (X + Y + Z) / √3

    Parameters
    ----------
    X : jnp.ndarray
        X-channel TDI time series
    Y : jnp.ndarray
        Y-channel TDI time series
    Z : jnp.ndarray
        Z-channel TDI time series

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Tuple of (A, E, T) channels

    Notes
    -----
    This transformation is based on the lisatools implementation and provides:
    - A: Antisymmetric combination, sensitive to gravitational waves
    - E: Equal-arm combination, sensitive to gravitational waves
    - T: Symmetric combination, primarily sensitive to laser frequency noise
    """
    A = (Z - X) / jnp.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / jnp.sqrt(6.0)
    T = (X + Y + Z) / jnp.sqrt(3.0)

    return A, E, T
