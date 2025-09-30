"""LISA response function computations using JAX."""

from typing import Tuple

import jax
import jax.numpy as jnp

from .interpolation import (
    lagrangian_interpolation,
    prepare_interpolation_coefficients,
)
from .tdi import TDIProcessor
from .utils import (
    dot_product_1d,
    get_basis_vecs,
    normalize_vector,
    xi_projections,
    xyz_to_aet,
)

# Physical constants
C_inv = 3.3356409519815204e-09  # Inverse speed of light in s/m
NLINKS = 6  # Number of LISA links


def _compute_single_link_response(
    input_waveform: jnp.ndarray,
    t_data: jnp.ndarray,
    k: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    x0_array: jnp.ndarray,
    x1_array: jnp.ndarray,
    L_array: jnp.ndarray,
    t_orbit: jnp.ndarray,  # Added: time points for orbital data
    num_pts: int,
    start_ind: int,
    sampling_frequency: float,
    A_coeffs: jnp.ndarray,
    E_coeffs: jnp.ndarray,
    deps: float,
) -> jnp.ndarray:
    """Compute response for a single LISA link (vectorized)."""

    # Get all time points we'll process
    time_indices = jnp.arange(num_pts) + start_ind
    t_values = t_data[time_indices]

    # Vectorized spacecraft position interpolation for all time points
    def interp_positions(t_val):
        x0 = jax.vmap(jnp.interp, in_axes=(None, None, 0))(
            t_val, t_orbit, x0_array.T
        ).T  # Receiver
        x1 = jax.vmap(jnp.interp, in_axes=(None, None, 0))(
            t_val, t_orbit, x1_array.T
        ).T  # Emitter
        L = jnp.interp(t_val, t_orbit, L_array)  # Light travel time
        return x0, x1, L

    # Apply to all time values at once
    x0_all, x1_all, L_all = jax.vmap(interp_positions)(t_values)

    # Vectorized link vector computation
    link_vecs = x0_all - x1_all
    n_all = jax.vmap(normalize_vector)(link_vecs)

    # Vectorized xi projections
    xi_p_all, xi_c_all = jax.vmap(xi_projections, in_axes=(None, None, 0))(
        u, v, n_all
    )

    # Vectorized dot products
    k_dot_n_all = jax.vmap(dot_product_1d, in_axes=(None, 0))(k, n_all)
    k_dot_x0_all = jax.vmap(dot_product_1d, in_axes=(None, 0))(k, x0_all)
    k_dot_x1_all = jax.vmap(dot_product_1d, in_axes=(None, 0))(k, x1_all)

    # Vectorized delay calculations
    delay0_all = t_values - k_dot_x0_all * C_inv
    delay1_all = t_values - L_all - k_dot_x1_all * C_inv

    # Vectorized integer delays and fractions
    integer_delay0_all = (
        jnp.ceil(delay0_all * sampling_frequency).astype(int) - 1
    )
    fraction0_all = 1.0 + integer_delay0_all - delay0_all * sampling_frequency

    integer_delay1_all = (
        jnp.ceil(delay1_all * sampling_frequency).astype(int) - 1
    )
    fraction1_all = 1.0 + integer_delay1_all - delay1_all * sampling_frequency

    # Vectorized Lagrangian interpolation
    def interpolate_single(int_delay, fraction):
        return lagrangian_interpolation(
            input_waveform, int_delay, fraction, A_coeffs, E_coeffs, deps
        )

    hp_del0_all, hc_del0_all = jax.vmap(interpolate_single)(
        integer_delay0_all, fraction0_all
    )
    hp_del1_all, hc_del1_all = jax.vmap(interpolate_single)(
        integer_delay1_all, fraction1_all
    )

    # Vectorized final computation
    pre_factor_all = 1.0 / (1.0 - k_dot_n_all)
    large_factor_all = (hp_del0_all - hp_del1_all) * xi_p_all + (
        hc_del0_all - hc_del1_all
    ) * xi_c_all

    return pre_factor_all * large_factor_all


class LISAResponse:
    """LISA gravitational wave response function computation.

    This class computes the response of the LISA detector to gravitational
    waves, including projections onto individual arms and TDI combinations.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency in Hz
    num_pts : int
        Number of output points
    order : int, optional
        Order of Lagrangian interpolation, by default 25
    """

    def __init__(
        self, sampling_frequency: float, num_pts: int, order: int = 25
    ):
        """Initialize LISA response calculator.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz
        num_pts : int
            Number of output points
        order : int, optional
            Order of Lagrangian interpolation, by default 25
        """
        self.sampling_frequency = sampling_frequency
        self.dt = 1.0 / sampling_frequency
        self.num_pts = num_pts
        self.order = order
        self.buffer_integer = order * 2 + 1

        # Prepare interpolation coefficients
        coeffs = prepare_interpolation_coefficients(order)
        self.A_coeffs, self.E_coeffs, self.deps = coeffs

        # Initialize TDI processor
        self.tdi_processor = TDIProcessor(sampling_frequency, order)
        self._compute_single_link_response_jit = jax.jit(
            _compute_single_link_response,
            static_argnums=(9,),
        )

    def get_projections(
        self,
        input_waveform: jnp.ndarray,
        lam: float,
        beta: float,
        orbits_data: dict,
        t0: float = 10000.0,
    ) -> jnp.ndarray:
        """Compute projections of GW signal onto LISA constellation.

        Parameters
        ----------
        input_waveform : jnp.ndarray
            Complex waveform h_+ + i*h_x
        lam : float
            Ecliptic longitude in radians
        beta : float
            Ecliptic latitude in radians
        orbits_data : dict
            Dictionary containing orbital information
        t0 : float, optional
            Start time buffer, by default 10000.0

        Returns
        -------
        jnp.ndarray
            Array of shape (6, num_pts) with projections for each link
        """
        # Get basis vectors
        u, v, k = get_basis_vecs(lam, beta)

        # Buffer calculations - match FLR exactly
        tdi_start_ind = int(t0 / self.dt)

        # FLR has two different buffers:
        # 1. check_tdi_buffer: used for projections_start_ind calculation
        check_tdi_buffer = (
            int(100.0 * self.sampling_frequency) + 4 * self.order
        )

        # 2. projection_buffer: used for safety checks
        # Calculate max spacecraft distance like FLR does
        max_sc_distance = 0.0
        for sc_pos in orbits_data["positions"]:
            distances = jnp.sqrt(jnp.sum(sc_pos * sc_pos, axis=1))
            max_sc_distance = max(max_sc_distance, float(jnp.max(distances)))

        projection_buffer = int(max_sc_distance * C_inv) + 4 * self.order

        # Use check_tdi_buffer for projections_start_ind (like FLR)
        projections_start_ind = tdi_start_ind - 2 * check_tdi_buffer

        # Fix time array to align projections_start_ind with t=0
        # This matches FLR behavior and fixes scaling issues
        t_data = (
            jnp.arange(len(input_waveform)) - projections_start_ind
        ) * self.dt
        # Trim time and waveform data to match orbital data
        # Match FLR behavior
        if t_data.max() > orbits_data["time"].max():
            max_ind = jnp.where(orbits_data["time"] >= t_data.max())[0][-1]
            t_data = t_data[:max_ind]
            input_waveform = input_waveform[:max_ind]

        if projections_start_ind < projection_buffer:
            raise ValueError("Need to increase t0. Buffer not large enough.")

        # Prepare output array
        y_gw = jnp.zeros((NLINKS, self.num_pts))

        # Link spacecraft mappings (LISA convention)
        # Links: 12, 21, 13, 31, 23, 32 -> indices 0,1,2,3,4,5
        sc_pairs = [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]

        # Pre-organize orbital data for vectorized processing
        all_x0_arrays = []
        all_x1_arrays = []
        all_L_arrays = []

        for link_idx in range(NLINKS):
            sc0, sc1 = sc_pairs[link_idx]
            all_x0_arrays.append(orbits_data["positions"][sc0 - 1])
            all_x1_arrays.append(orbits_data["positions"][sc1 - 1])
            all_L_arrays.append(orbits_data["light_travel_times"][link_idx])

        # Stack arrays for vectorized processing
        x0_stack = jnp.stack(all_x0_arrays)  # (6, time, 3)
        x1_stack = jnp.stack(all_x1_arrays)  # (6, time, 3)
        L_stack = jnp.stack(all_L_arrays)  # (6, time)

        # Vectorized link processing function
        def process_single_link_vectorized(x0_array, x1_array, L_array):
            return self._compute_single_link_response_jit(
                input_waveform,
                t_data,
                k,
                u,
                v,
                x0_array,
                x1_array,
                L_array,
                orbits_data["time"],
                self.num_pts,
                projections_start_ind,
                self.sampling_frequency,
                self.A_coeffs,
                self.E_coeffs,
                self.deps,
            )

        # Apply vectorized processing over all links
        y_gw = jax.vmap(process_single_link_vectorized)(
            x0_stack, x1_stack, L_stack
        )

        # Apply garbage removal to projections to match FLR behavior
        # Zero out the projection buffer regions at start and end
        y_gw = y_gw.at[:, :projections_start_ind].set(0.0)
        end_garbage_ind = -projections_start_ind + self.num_pts
        if end_garbage_ind < y_gw.shape[1]:
            y_gw = y_gw.at[:, end_garbage_ind:].set(0.0)

        return y_gw

    def get_tdi_channels(
        self,
        projections: jnp.ndarray,
        orbits_data: dict,
        tdi_type: str = "1st generation",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute TDI channels from projections.

        Parameters
        ----------
        projections : jnp.ndarray
            Array of shape (6, N) with link projections
        orbits_data : dict
            Dictionary with orbital information
        tdi_type : str, optional
            Type of TDI ("1st generation" or "2nd generation"), by default "1st generation"

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Tuple of (X, Y, Z) TDI channels
        """
        proj_length = projections.shape[1]
        t_arr = jnp.arange(proj_length) * self.dt

        # Pass orbital data directly for on-the-fly interpolation
        # This matches fastlisaresponse approach where light travel times
        # are interpolated during TDI computation, not pre-computed
        orbital_time = orbits_data["time"]
        orbital_ltt = orbits_data["light_travel_times"]

        return self.tdi_processor.compute_tdi_channels(
            projections, t_arr, orbital_time, orbital_ltt, tdi_type
        )

    def compute_response(
        self,
        input_waveform: jnp.ndarray,
        lam: float,
        beta: float,
        orbits_data: dict,
        t0: float = 10000.0,
        tdi_type: str = "1st generation",
        tdi_channels: str = "XYZ",
        return_projections: bool = False,
    ) -> Tuple[jnp.ndarray, ...]:
        """Complete LISA response computation.

        Parameters
        ----------
        input_waveform : jnp.ndarray
            Complex waveform h_+ + i*h_x
        lam : float
            Ecliptic longitude in radians
        beta : float
            Ecliptic latitude in radians
        orbits_data : dict
            Dictionary containing orbital information
        t0 : float, optional
            Start time buffer, by default 10000.0
        tdi_type : str, optional
            Type of TDI, by default "1st generation"
        tdi_channels : str, optional
            TDI channel format ("XYZ" or "AET"), by default "XYZ"
        return_projections : bool, optional
            Whether to return individual projections, by default False

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            TDI channels in requested format and optionally projections
        """
        projections = self.get_projections(
            input_waveform, lam, beta, orbits_data, t0
        )

        X, Y, Z = self.get_tdi_channels(projections, orbits_data, tdi_type)

        # Apply garbage removal like fastlisaresponse does
        # Remove t0/dt samples from beginning and end
        tdi_start_ind = int(t0 / self.dt)
        X_trimmed = X[tdi_start_ind:-tdi_start_ind]
        Y_trimmed = Y[tdi_start_ind:-tdi_start_ind]
        Z_trimmed = Z[tdi_start_ind:-tdi_start_ind]

        # Convert to requested channel format
        if tdi_channels.upper() == "XYZ":
            channels = (X_trimmed, Y_trimmed, Z_trimmed)
        elif tdi_channels.upper() == "AET":
            A, E, T = xyz_to_aet(X_trimmed, Y_trimmed, Z_trimmed)
            channels = (A, E, T)
        else:
            raise ValueError("tdi_channels must be 'XYZ' or 'AET'")

        if return_projections:
            return channels, projections
        else:
            return channels

    def __call__(self, *args, **kwds):
        return self.compute_response(*args, **kwds)
