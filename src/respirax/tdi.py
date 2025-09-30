"""TDI (Time-Delay Interferometry) computations for LISA."""

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

from .interpolation import (
    lagrangian_interpolation_real,
    prepare_interpolation_coefficients,
)

FIRST_GEN_COMBS: List[Dict] = [
    {"link": 13, "links_for_delay": [], "sign": +1},
    {"link": 31, "links_for_delay": [13], "sign": +1},
    {"link": 12, "links_for_delay": [13, 31], "sign": +1},
    {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
    {"link": 12, "links_for_delay": [], "sign": -1},
    {"link": 21, "links_for_delay": [12], "sign": -1},
    {"link": 13, "links_for_delay": [12, 21], "sign": -1},
    {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
]

SECOND_GEN_EXTRA: List[Dict] = [
    {"link": 12, "links_for_delay": [13, 31, 12, 21], "sign": +1},
    {"link": 21, "links_for_delay": [13, 31, 12, 21, 12], "sign": +1},
    {"link": 13, "links_for_delay": [13, 31, 12, 21, 12, 21], "sign": +1},
    {"link": 31, "links_for_delay": [13, 31, 12, 21, 12, 21, 13], "sign": +1},
    {"link": 13, "links_for_delay": [12, 21, 13, 31], "sign": -1},
    {"link": 31, "links_for_delay": [12, 21, 13, 31, 13], "sign": -1},
    {"link": 12, "links_for_delay": [12, 21, 13, 31, 13, 31], "sign": -1},
    {"link": 21, "links_for_delay": [12, 21, 13, 31, 13, 31, 12], "sign": -1},
]


class TDIProcessor:
    """TDI processor for LISA data.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency in Hz
    order : int, optional
        Order of Lagrangian interpolation, by default 25
    """

    def __init__(self, sampling_frequency: float, order: int = 25):
        self.sampling_frequency = sampling_frequency
        self.dt = 1.0 / sampling_frequency
        self.order = order
        self.buffer_integer = order * 2 + 1

        # Prepare interpolation coefficients
        coeffs = prepare_interpolation_coefficients(order)
        self.A_coeffs, self.E_coeffs, self.deps = coeffs

        # Standard LISA link ordering
        self.link_indices = {12: 0, 21: 5, 13: 3, 31: 2, 23: 1, 32: 4}

    def setup_tdi_combinations(
        self, tdi_type: str = "1st generation"
    ) -> List[Dict]:
        """Return TDI combination configurations.

        Parameters
        ----------
        tdi_type : str, optional
            Type of TDI ("1st generation" or "2nd generation"), by default "1st generation"

        Returns
        -------
        List[Dict]
            List of TDI combination dictionaries
        """
        if tdi_type == "1st generation":
            return FIRST_GEN_COMBS
        elif tdi_type == "2nd generation":
            return FIRST_GEN_COMBS + SECOND_GEN_EXTRA
        else:
            raise ValueError(
                "tdi_type must be '1st generation' or '2nd generation'"
            )

    @staticmethod
    def cyclic_permutation(link: int, permutation: int) -> int:
        """Apply cyclic permutation (1→2→3) to a link index.

        Parameters
        ----------
        link : int
            Link index (e.g., 13, 21, etc.)
        permutation : int
            Permutation index (0, 1, or 2)

        Returns
        -------
        int
            Permuted link index
        """
        i, j = divmod(link, 10)  # e.g. 13 → (1, 3)
        return 10 * ((i - 1 + permutation) % 3 + 1) + (
            (j - 1 + permutation) % 3 + 1
        )

    def compute_tdi_channels(
        self,
        input_links: jnp.ndarray,  # shape (6, N)
        t_arr: jnp.ndarray,  # shape (N,)
        orbital_time: jnp.ndarray,  # orbital time array
        orbital_ltt: jnp.ndarray,  # orbital light travel times (6, M)
        tdi_type: str = "1st generation",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute TDI X, Y, Z channels.

        Parameters
        ----------
        input_links : (6, N) jnp.ndarray
            Link projections in standard LISA link order.
        t_arr : (N,) jnp.ndarray
            Time stamps.
        orbital_time : (M,) jnp.ndarray
            Orbital time array for interpolation.
        orbital_ltt : (6, M) jnp.ndarray
            Orbital light travel times for interpolation.
        tdi_type : str, optional
            Type of TDI, by default "1st generation"

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Time series of the TDI channels (X, Y, Z)
        """
        num_points = t_arr.shape[0]
        tdi_combinations = self.setup_tdi_combinations(tdi_type)

        assert input_links.shape == (6, num_points), (
            f"input_links shape {input_links.shape}, "
            f"expected (6, {num_points})"
        )
        assert orbital_ltt.shape[0] == 6, (
            f"orbital_ltt first dim {orbital_ltt.shape[0]}, expected 6"
        )

        def process_term_optimized(tdi_term, permutation):
            """Process a single TDI term with JAX optimizations."""
            base_link = self.cyclic_permutation(tdi_term["link"], permutation)
            delay_links = [
                self.cyclic_permutation(link, permutation)
                for link in tdi_term["links_for_delay"]
            ]
            sign = tdi_term["sign"]

            # Start with original time array
            delays = t_arr.copy()

            # Vectorized light travel time subtraction
            if delay_links:
                # Get all link indices at once
                link_indices = jnp.array(
                    [self.link_indices[link] for link in delay_links]
                )

                # Vectorized interpolation for all delay links
                def interp_ltt_single(link_idx):
                    return jnp.interp(
                        t_arr, orbital_time, orbital_ltt[link_idx]
                    )

                lt_values = jax.vmap(interp_ltt_single)(link_indices)
                total_lt = jnp.sum(lt_values, axis=0)
                delays = delays - total_lt

            base_link_idx = self.link_indices[base_link]
            base_data = input_links[base_link_idx]

            # Vectorized delay computation
            int_delays = (
                jnp.ceil(delays * self.sampling_frequency).astype(int) - 1
            )
            fractions = 1.0 + int_delays - delays * self.sampling_frequency

            # Vectorized interpolation
            def interp_one(int_delay, fraction):
                return lagrangian_interpolation_real(
                    base_data,
                    int_delay,
                    fraction,
                    self.A_coeffs,
                    self.E_coeffs,
                    self.deps,
                )

            results = jax.vmap(interp_one)(int_delays, fractions)
            return sign * results

        # Process all three permutations
        channels = []
        for permutation in range(3):
            channel_sum = jnp.zeros(num_points)
            for tdi_term in tdi_combinations:
                term_result = process_term_optimized(tdi_term, permutation)
                channel_sum = channel_sum + term_result
            channels.append(channel_sum)

        return tuple(channels)
