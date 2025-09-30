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
    get_basis_vecs,
    normalize_vector,
    xi_projections,
    xyz_to_aet,
)

# Physical constants
C_inv = 3.3356409519815204e-09  # Inverse speed of light in s/m
NLINKS = 6  # Number of LISA links


@jax.jit
def _compute_single_link_response(
    input_waveform: jnp.ndarray,
    t_values: jnp.ndarray,
    k: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    x0_all: jnp.ndarray,
    x1_all: jnp.ndarray,
    L_all: jnp.ndarray,
    n_all: jnp.ndarray,
    sampling_frequency: float,
    A_coeffs: jnp.ndarray,
    E_coeffs: jnp.ndarray,
    deps: float,
) -> jnp.ndarray:
    """Compute response for a single LISA link (vectorized)."""

    # Vectorized xi projections
    xi_p_all, xi_c_all = jax.vmap(xi_projections, in_axes=(None, None, 0))(
        u, v, n_all
    )

    # Vectorized dot products
    k_dot_n_all = n_all @ k  # (N,)
    k_dot_x0_all = x0_all @ k  # (N,)
    k_dot_x1_all = x1_all @ k  # (N,)

    # Vectorized Lagrangian interpolation
    def interpolate_single(int_delay, fraction):
        return lagrangian_interpolation(
            input_waveform, int_delay, fraction, A_coeffs, E_coeffs, deps
        )

    def complex_delays(delay):
        delay_all = t_values - delay
        int_delay_all = (
            jnp.ceil(delay_all * sampling_frequency).astype(int) - 1
        )
        fraction_all = 1.0 + int_delay_all - delay_all * sampling_frequency
        hp_del_all, hc_del_all = jax.vmap(interpolate_single)(
            int_delay_all, fraction_all
        )
        return hp_del_all, hc_del_all

    hp_del0_all, hc_del0_all = complex_delays(k_dot_x0_all * C_inv)
    hp_del1_all, hc_del1_all = complex_delays(L_all + k_dot_x1_all * C_inv)

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
        self,
        sampling_frequency: float,
        num_pts: int,
        orbits_data,
        order: int = 25,
        t0: float = 10000.0,
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
        self.t0 = t0

        self.buffer_integer = order * 2 + 1

        # Prepare interpolation coefficients
        coeffs = prepare_interpolation_coefficients(order)
        self.A_coeffs, self.E_coeffs, self.deps = coeffs

        # Initialize TDI processor
        self.tdi_processor = TDIProcessor(sampling_frequency, order)

        self.orbits_data = orbits_data

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
        self.x0_stack = jnp.stack(all_x0_arrays)  # (6, time, 3)
        self.x1_stack = jnp.stack(all_x1_arrays)  # (6, time, 3)
        self.L_stack = jnp.stack(all_L_arrays)  # (6, time)

        # Buffer calculations - match FLR exactly
        self.tdi_start_ind = int(t0 / self.dt)

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
        self.projections_start_ind = self.tdi_start_ind - 2 * check_tdi_buffer

        self.t_data = (
            jnp.arange(self.num_pts) - self.projections_start_ind
        ) * self.dt

        self.orbits_t = orbits_data["time"]
        self.orbits_tmax = orbits_data["time"].max()

        if self.projections_start_ind < projection_buffer:
            raise ValueError(
                "Need to increase t0. Buffer not large enough."
            )  # This error message needs made better

        if self.t_data.max() > self.orbits_tmax:
            raise ValueError(
                "Orbit data time range too short for requested num_pts and t0."
            )

        time_indices = jnp.arange(num_pts) + self.projections_start_ind
        self.t_values = self.t_data[time_indices]

        def interp_wrap(t_val, x0_array, x1_array, L_array):
            def interp_positions(t_val):
                x0 = jax.vmap(jnp.interp, in_axes=(None, None, 0))(
                    t_val, self.orbits_t, x0_array.T
                ).T  # Receiver
                x1 = jax.vmap(jnp.interp, in_axes=(None, None, 0))(
                    t_val, self.orbits_t, x1_array.T
                ).T  # Emitter
                L = jnp.interp(
                    t_val, self.orbits_t, L_array
                )  # Light travel time
                return x0, x1, L

            x0_all, x1_all, L_all = jax.vmap(interp_positions)(t_val)
            link_vecs = x0_all - x1_all
            n_all = jax.vmap(normalize_vector)(link_vecs)

            return x0_all, x1_all, L_all, n_all

        self.x0_array, self.x1_array, self.L_array, self.n_array = jax.vmap(
            interp_wrap, in_axes=(None, 0, 0, 0)
        )(self.t_values, self.x0_stack, self.x1_stack, self.L_stack)

    def get_projections(
        self,
        input_waveform: jnp.ndarray,
        lam: float,
        beta: float,
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

        # Prepare output array
        y_gw = jnp.zeros((NLINKS, self.num_pts))

        # Vectorized link processing function
        def process_single_link_vectorized(
            x0_array, x1_array, L_array, n_array
        ):
            return _compute_single_link_response(
                input_waveform,
                self.t_values,
                k,
                u,
                v,
                x0_array,
                x1_array,
                L_array,
                n_array,
                self.sampling_frequency,
                self.A_coeffs,
                self.E_coeffs,
                self.deps,
            )

        # Apply vectorized processing over all links
        y_gw = jax.vmap(process_single_link_vectorized)(
            self.x0_array, self.x1_array, self.L_array, self.n_array
        )

        # Apply garbage removal to projections to match FLR behavior
        # Zero out the projection buffer regions at start and end
        y_gw = y_gw.at[:, : self.projections_start_ind].set(0.0)
        end_garbage_ind = -self.projections_start_ind + self.num_pts
        if end_garbage_ind < y_gw.shape[1]:
            y_gw = y_gw.at[:, end_garbage_ind:].set(0.0)

        return y_gw

    def get_tdi_channels(
        self,
        projections: jnp.ndarray,
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
        orbital_time = self.orbits_data["time"]
        orbital_ltt = self.orbits_data["light_travel_times"]

        return self.tdi_processor.compute_tdi_channels(
            projections, t_arr, orbital_time, orbital_ltt, tdi_type
        )

    def compute_response(
        self,
        input_waveform: jnp.ndarray,
        lam: float,
        beta: float,
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
            input_waveform,
            lam,
            beta,
        )

        X, Y, Z = self.get_tdi_channels(projections, tdi_type)

        # Apply garbage removal like fastlisaresponse does
        # Remove t0/dt samples from beginning and end
        X_trimmed = X[self.tdi_start_ind : -self.tdi_start_ind]
        Y_trimmed = Y[self.tdi_start_ind : -self.tdi_start_ind]
        Z_trimmed = Z[self.tdi_start_ind : -self.tdi_start_ind]

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
