"""Lagrangian interpolation functions for LISA response computations."""

from typing import Tuple

import jax
import jax.numpy as jnp

from .utils import generate_factorial_array


def prepare_interpolation_coefficients(
    order: int, num_A: int = 1001
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Prepare coefficients for Lagrangian interpolation.

    Parameters
    ----------
    order: int
        Order of interpolation
    num_A: int
        Number of points for A coefficient interpolation

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, float]
        Tuple of (A_coeffs, E_coeffs, deps) where:
        - A_coeffs: Array of A coefficient values for interpolation
        - E_coeffs: Array of E coefficient values
        - deps: Spacing between epsilon values
    """
    factorials = generate_factorial_array(order + 10)
    deps = 1.0 / (num_A - 1)
    eps = jnp.arange(num_A) * deps

    h = int((order + 1) / 2)
    denominator = factorials[h - 1] * factorials[h]

    # Prepare A coefficients
    def compute_A(eps_val):
        A = 1.0
        for i in range(1, h):
            A *= (i + eps_val) * (i + 1 - eps_val)
        return A / denominator

    A_coeffs = jax.vmap(compute_A)(eps)

    # Prepare E coefficients - vectorized
    j_values = jnp.arange(1, h)
    first_terms = factorials[h - 1] / factorials[h - 1 - j_values]
    second_terms = factorials[h] / factorials[h + j_values]
    E_coeffs = first_terms * second_terms * ((-1.0) ** j_values)

    return A_coeffs, E_coeffs, deps


@jax.jit
def lagrangian_interpolation(
    input_array: jnp.ndarray,
    delay_index: int,
    fraction: float,
    A_coeffs: jnp.ndarray,
    E_coeffs: jnp.ndarray,
    deps: float,
    start_input_ind: int = 0,
) -> Tuple[float, float]:
    """Perform Lagrangian interpolation for complex input.

    This function implements the Lagrangian interpolation scheme used in the
    original CUDA code for interpolating gravitational wave data.

    Args:
        input_array: Complex input array to interpolate from
        delay_index: Integer part of the delay
        fraction: Fractional part of delay interpolation
        A_coeffs: Pre-computed A coefficient array
        E_coeffs: Pre-computed E coefficient array
        deps: Spacing between epsilon values
        start_input_ind: Starting index offset in input array

    Returns:
        Tuple of (hp_interpolated, hc_interpolated) for plus and cross
        polarizations
    """
    half_order = len(E_coeffs) + 1

    # Get A coefficient via linear interpolation
    eps_index = jnp.floor(fraction / deps).astype(int)
    eps_index = jnp.clip(eps_index, 0, len(A_coeffs) - 2)
    frac = (fraction - eps_index * deps) / deps
    A = A_coeffs[eps_index] * (1.0 - frac) + A_coeffs[eps_index + 1] * frac

    B = 1.0 - fraction
    C = fraction
    D = fraction * (1.0 - fraction)

    # Main interpolation loop - vectorized
    j_values = jnp.arange(1, half_order)
    E_values = E_coeffs[j_values - 1]
    F_values = j_values + fraction
    G_values = j_values + (1 - fraction)

    # Get input indices with boundary checks
    up_indices = jnp.clip(
        delay_index + 1 + j_values - start_input_ind, 0, len(input_array) - 1
    )
    down_indices = jnp.clip(
        delay_index - j_values - start_input_ind, 0, len(input_array) - 1
    )

    temp_up = input_array[up_indices]
    temp_down = input_array[down_indices]

    # Vectorized sum computation
    hp_terms = temp_up.real / F_values + temp_down.real / G_values
    hc_terms = temp_up.imag / F_values + temp_down.imag / G_values
    sum_hp = jnp.sum(E_values * hp_terms)
    sum_hc = jnp.sum(E_values * hc_terms)

    # Final terms
    up_idx = jnp.clip(
        delay_index + 1 - start_input_ind, 0, len(input_array) - 1
    )
    down_idx = jnp.clip(delay_index - start_input_ind, 0, len(input_array) - 1)

    temp_up = input_array[up_idx]
    temp_down = input_array[down_idx]

    result_hp = A * (B * temp_up.real + C * temp_down.real + D * sum_hp)
    result_hc = A * (B * temp_up.imag + C * temp_down.imag + D * sum_hc)

    return result_hp, result_hc


@jax.jit
def lagrangian_interpolation_real(
    input_array: jnp.ndarray,
    delay_index: int,
    fraction: float,
    A_coeffs: jnp.ndarray,
    E_coeffs: jnp.ndarray,
    deps: float,
    start_input_ind: int = 0,
) -> float:
    """Perform Lagrangian interpolation for real input.

    Args:
        input_array: Real input array to interpolate from
        delay_index: Integer part of the delay
        fraction: Fractional part of delay interpolation
        A_coeffs: Pre-computed A coefficient array
        E_coeffs: Pre-computed E coefficient array
        deps: Spacing between epsilon values
        start_input_ind: Starting index offset in input array

    Returns:
        Interpolated real value
    """
    half_order = len(E_coeffs) + 1

    # Get A coefficient via linear interpolation
    eps_index = jnp.floor(fraction / deps).astype(int)
    eps_index = jnp.clip(eps_index, 0, len(A_coeffs) - 2)
    frac = (fraction - eps_index * deps) / deps
    A = A_coeffs[eps_index] * (1.0 - frac) + A_coeffs[eps_index + 1] * frac

    B = 1.0 - fraction
    C = fraction
    D = fraction * (1.0 - fraction)

    # Main interpolation loop - vectorized
    j_values = jnp.arange(1, half_order)
    E_values = E_coeffs[j_values - 1]
    F_values = j_values + fraction
    G_values = j_values + (1 - fraction)

    # Get input indices with boundary checks
    up_indices = jnp.clip(
        delay_index + 1 + j_values - start_input_ind, 0, len(input_array) - 1
    )
    down_indices = jnp.clip(
        delay_index - j_values - start_input_ind, 0, len(input_array) - 1
    )

    temp_up = input_array[up_indices]
    temp_down = input_array[down_indices]

    # Vectorized sum computation
    sum_val = jnp.sum(E_values * (temp_up / F_values + temp_down / G_values))

    # Final terms
    up_idx = jnp.clip(
        delay_index + 1 - start_input_ind, 0, len(input_array) - 1
    )
    down_idx = jnp.clip(delay_index - start_input_ind, 0, len(input_array) - 1)

    temp_up = input_array[up_idx]
    temp_down = input_array[down_idx]

    result = A * (B * temp_up + C * temp_down + D * sum_val)

    return result
