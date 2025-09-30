"""LISA response function computations using JAX."""

import jax
import jax.numpy as jnp
from typing import Tuple
from .utils import (
    get_basis_vecs,
    xi_projections,
    dot_product_1d,
    normalize_vector,
    xyz_to_aet,
)
from .interpolation import (
    lagrangian_interpolation,
    prepare_interpolation_coefficients,
)
from .tdi import TDIProcessor

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
    num_pts: int,
    start_ind: int,
    sampling_frequency: float,
    A_coeffs: jnp.ndarray,
    E_coeffs: jnp.ndarray,
    deps: float,
) -> jnp.ndarray:
    """Compute response for a single LISA link (JAX-compatible)."""

    def body_fun(i, output):
        j = i + start_ind

        t = t_data[j]
        x0 = x0_array[j]  # Receiver
        x1 = x1_array[j]  # Emitter

        link_vec = x0 - x1
        n = normalize_vector(link_vec)

        L = L_array[j]

        xi_p, xi_c = xi_projections(u, v, n)

        k_dot_n = dot_product_1d(k, n)
        k_dot_x0 = dot_product_1d(k, x0)
        k_dot_x1 = dot_product_1d(k, x1)

        delay0 = t - k_dot_x0 * C_inv
        delay1 = t - L - k_dot_x1 * C_inv

        integer_delay0 = jnp.ceil(delay0 * sampling_frequency).astype(int) - 1
        fraction0 = 1.0 + integer_delay0 - delay0 * sampling_frequency

        integer_delay1 = jnp.ceil(delay1 * sampling_frequency).astype(int) - 1
        fraction1 = 1.0 + integer_delay1 - delay1 * sampling_frequency

        hp_del0, hc_del0 = lagrangian_interpolation(
            input_waveform, integer_delay0, fraction0, A_coeffs, E_coeffs, deps
        )
        hp_del1, hc_del1 = lagrangian_interpolation(
            input_waveform, integer_delay1, fraction1, A_coeffs, E_coeffs, deps
        )

        pre_factor = 1.0 / (1.0 - k_dot_n)
        large_factor = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c

        return output.at[i].set(pre_factor * large_factor)

    output0 = jnp.zeros(num_pts)
    return jax.lax.fori_loop(0, num_pts, body_fun, output0)


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
            static_argnums=(8,),
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

        # Setup time array
        t_data = jnp.arange(len(input_waveform)) * self.dt

        # Buffer calculations
        tdi_start_ind = int(t0 / self.dt)
        projection_buffer = (
            int(100.0 * self.sampling_frequency) + 4 * self.order
        )
        projections_start_ind = tdi_start_ind - 2 * projection_buffer

        if projections_start_ind < projection_buffer:
            raise ValueError("Need to increase t0. Buffer not large enough.")

        # Prepare output array
        y_gw = jnp.zeros((NLINKS, self.num_pts))

        # Link spacecraft mappings (LISA convention)
        # Links: 12, 21, 13, 31, 23, 32 -> indices 0,1,2,3,4,5
        sc_pairs = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

        # Process each link
        for link_idx in range(NLINKS):
            sc0, sc1 = sc_pairs[link_idx]

            # Get spacecraft positions for this link
            x0_array = orbits_data["positions"][sc0 - 1]  # 0-indexed
            x1_array = orbits_data["positions"][sc1 - 1]
            L_array = orbits_data["light_travel_times"][link_idx]

            link_projection = self._compute_single_link_response_jit(
                input_waveform,
                t_data,
                k,
                u,
                v,
                x0_array,
                x1_array,
                L_array,
                self.num_pts,
                projections_start_ind,
                self.sampling_frequency,
                self.A_coeffs,
                self.E_coeffs,
                self.deps,
            )

            y_gw = y_gw.at[link_idx].set(link_projection)

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
