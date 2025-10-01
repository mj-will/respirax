"""TDI (Time-Delay Interferometry) computations for LISA.

The implementation supports both 1st and 2nd generation TDI combinations.
"""

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

from .interpolation import (
    lagrangian_interpolation_real,
    prepare_interpolation_coefficients,
)

# TDI combination definitions for LISA interferometer links
# Links are numbered as: 12, 13, 21, 23, 31, 32 representing connections
# between spacecraft 1, 2, and 3

# First generation TDI combinations - the basic set of 8 terms
# Each term specifies:
# - link: the primary interferometer link to read data from
# - links_for_delay: chain of links whose light travel times create the delay
# - sign: +1 or -1 for positive or negative contribution to the combination
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

# Second generation TDI
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
    """High-performance TDI processor for LISA gravitational wave data.

    This class implements efficient Time-Delay Interferometry computation using
    JAX for vectorization and JIT compilation. It precomputes interpolation
    coefficients and TDI term structures for optimal performance.

    The processor supports both 1st and 2nd generation TDI, which differ in
    the number of terms used for laser noise cancellation:
    - 1st generation: 8 terms, faster computation
    - 2nd generation: 16 terms, better noise cancellation

    Parameters
    ----------
    sampling_frequency : float
        Data sampling frequency in Hz. Typical LISA values are 0.1-1 Hz.
    order : int, optional
        Order of Lagrangian interpolation for time-delay operations.
        Higher orders provide better accuracy but increase computation cost.
        Default is 25, which provides excellent accuracy for LISA.

    Attributes
    ----------
    sampling_frequency : float
        The data sampling frequency in Hz
    dt : float
        Sampling time interval (1/sampling_frequency)
    order : int
        Lagrangian interpolation order
    buffer_integer : int
        Buffer size for interpolation (2*order + 1)
    A_coeffs, E_coeffs, deps : jnp.ndarray
        Precomputed Lagrangian interpolation coefficients

    Examples
    --------
    >>> # Create processor for 0.1 Hz LISA data
    >>> processor = TDIProcessor(sampling_frequency=0.1, order=25)
    >>>
    >>> # Compute TDI channels from interferometer data
    >>> X, Y, Z = processor.compute_tdi_channels(
    ...     input_links, t_arr, orbital_time, orbital_ltt
    ... )
    """

    def __init__(self, sampling_frequency: float, order: int = 25):
        self.sampling_frequency = sampling_frequency
        self.dt = 1.0 / sampling_frequency  # Sampling time interval
        self.order = order
        self.buffer_integer = order * 2 + 1  # Buffer for interpolation

        # Precompute interpolation coefficients (fixed order)
        # These are used for all Lagrangian interpolations in TDI
        self.A_coeffs, self.E_coeffs, self.deps = (
            prepare_interpolation_coefficients(order)
        )

        # LISA link code to array index mapping
        # Map: 12→0, 23→1, 31→2, 13→3, 32→4, 21→5
        # This maps two-digit link codes to array indices for data access
        self._link_code_to_idx = {12: 0, 23: 1, 31: 2, 13: 3, 32: 4, 21: 5}

        # Precompute term tensors and kernels for both TDI generations
        # This front-loads computation for better runtime performance
        self._terms_first = self._build_terms(FIRST_GEN_COMBS)
        self._terms_second = self._build_terms(
            FIRST_GEN_COMBS + SECOND_GEN_EXTRA
        )

        # Create JIT-compiled kernels for optimal performance
        self._kernel_first = self._make_kernel(self._terms_first)
        self._kernel_second = self._make_kernel(self._terms_second)

    @staticmethod
    def _interp_uniform(x, xp0, dx, fp):
        """Linear interpolation for uniformly spaced grid.

        Performs linear interpolation on a uniform grid defined by
        xp = xp0 + dx*k, which is more efficient than general interpolation
        since grid spacing is constant.

        Parameters
        ----------
        x : jnp.ndarray
            Query points for interpolation
        xp0 : float
            Starting point of uniform grid
        dx : float
            Grid spacing
        fp : jnp.ndarray
            Function values at grid points

        Returns
        -------
        jnp.ndarray
            Interpolated values at query points
        """
        # Convert x coordinates to fractional grid indices
        k = (x - xp0) / dx
        i = jnp.floor(k).astype(jnp.int32)  # Integer part
        i = jnp.clip(i, 0, fp.size - 2)  # Ensure valid range
        frac = k - i  # Fractional part

        # Linear interpolation: f(x) = f[i]*(1-frac) + f[i+1]*frac
        return fp[i] * (1.0 - frac) + fp[i + 1] * frac

    @staticmethod
    def _cyclic_perm(links: jnp.ndarray, p: int):
        """Apply cyclic permutation (1→2→3→1) to LISA link codes.

        Parameters
        ----------
        links : jnp.ndarray
            Two-digit link codes (e.g., 12, 23, 31)
        p : int
            Permutation index: 0=identity, 1=(1→2→3→1), 2=(1→3→2→1)

        Returns
        -------
        jnp.ndarray
            Permuted link codes
        """
        # Extract spacecraft indices from two-digit codes
        i = links // 10  # First spacecraft (tens digit)
        j = links % 10  # Second spacecraft (ones digit)

        # Apply cyclic permutation: (1,2,3) → (2,3,1) → (3,1,2)
        ip = ((i - 1 + p) % 3) + 1
        jp = ((j - 1 + p) % 3) + 1

        # Reconstruct two-digit codes
        return 10 * ip + jp

    def _code_to_idx(self, links: jnp.ndarray):
        """Convert LISA link codes to array indices.

        Maps two-digit spacecraft link codes (12, 13, 21, 23, 31, 32)
        to array indices (0-5) for accessing interferometer data arrays.

        Parameters
        ----------
        links : jnp.ndarray
            Array of two-digit link codes

        Returns
        -------
        jnp.ndarray
            Array indices corresponding to link codes
        """
        # Initialize with invalid index (-1)
        idx = jnp.full_like(links, -1)

        # Map each known link code to its array index
        for code, k in self._link_code_to_idx.items():
            idx = jnp.where(links == code, k, idx)
        return idx

    def _pad_delays(self, delays_list: List[List[int]]):
        """Pad delay link lists to uniform length for vectorization.

        TDI terms have different numbers of delay links. This method
        pads all delay lists to the same length with invalid indices,
        and creates a mask to identify valid entries.

        Parameters
        ----------
        delays_list : List[List[int]]
            List of delay link lists, potentially different lengths

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Padded delay array and boolean mask for valid entries
        """
        # Find maximum length across all delay lists
        Lmax = max((len(d) for d in delays_list), default=0)

        # Pad each list with -1 (invalid index) to reach maximum length
        arr = []
        for d in delays_list:
            arr.append(d + [-1] * (Lmax - len(d)))

        # Create padded array and validity mask
        padded_array = jnp.array(arr, dtype=jnp.int32)
        validity_mask = jnp.array(
            [[v >= 0 for v in row] for row in arr], dtype=bool
        )

        return padded_array, validity_mask

    def _build_terms(self, combo_list: List[Dict]):
        """Build vectorized TDI term structure from combination list.

        Converts the human-readable TDI combination definitions into
        optimized array structures suitable for vectorized computation.
        This includes applying cyclic permutations and converting link
        codes to array indices.

        Parameters
        ----------
        combo_list : List[Dict]
            List of TDI combination dictionaries with keys:
            - 'link': primary link code
            - 'links_for_delay': list of delay link codes
            - 'sign': +1 or -1 for term sign

        Returns
        -------
        Dict
            Processed term structure with arrays:
            - signs: (T,) term signs
            - base_idx: (3,T) base link indices for 3 permutations
            - delay_idx: (3,T,L) delay link indices
            - delay_mask: (T,L) mask for valid delay entries
        """
        # Extract components from combination list
        base = jnp.array(
            [c["link"] for c in combo_list], dtype=jnp.int32
        )  # (T,) base links
        signs = jnp.array(
            [c["sign"] for c in combo_list], dtype=jnp.int32
        )  # (T,) signs

        # Pad delay lists to uniform length
        delays_raw = [c["links_for_delay"] for c in combo_list]
        delay_links, delay_mask = self._pad_delays(delays_raw)  # (T,L), (T,L)

        # Apply cyclic permutations to get all 3 TDI channels (X,Y,Z)
        base_perm = jnp.stack(
            [self._cyclic_perm(base, p) for p in (0, 1, 2)], axis=0
        )  # (3,T) base links for 3 permutations
        delay_perm = jnp.stack(
            [self._cyclic_perm(delay_links, p) for p in (0, 1, 2)], 0
        )  # (3,T,L) delay links for 3 permutations

        # Convert link codes to array indices
        base_idx = self._code_to_idx(base_perm)  # (3,T)
        delay_idx = self._code_to_idx(delay_perm)  # (3,T,L)

        return {
            "signs": signs,
            "base_idx": base_idx,
            "delay_idx": delay_idx,
            "delay_mask": delay_mask,
        }

    def _make_kernel(self, terms):
        """Create JIT-compiled kernel for TDI computation.

        Parameters
        ----------
        terms : Dict
            Processed TDI term structure from _build_terms

        Returns
        -------
        Callable
            JIT-compiled function for TDI computation
        """
        signs = terms["signs"]
        base_idx = terms["base_idx"]
        delay_idx = terms["delay_idx"]
        delay_mask0 = terms["delay_mask"]

        # Cache interpolation coefficients for kernel
        A, E, deps = self.A_coeffs, self.E_coeffs, self.deps
        fs = self.sampling_frequency

        def _kernel(input_links, t_arr, orbital_time, orbital_ltt):
            """Optimized TDI computation kernel.

            Parameters
            ----------
            input_links : jnp.ndarray, shape (6,N)
                Interferometer data for all 6 LISA links
            t_arr : jnp.ndarray, shape (N,)
                Time array for data points
            orbital_time : jnp.ndarray, shape (M,)
                Time grid for orbital light travel time data
            orbital_ltt : jnp.ndarray, shape (6,M)
                Orbital light travel times for all links

            Returns
            -------
            jnp.ndarray, shape (3,N)
                TDI channel data (X, Y, Z)
            """
            # Set up uniform grid parameters for efficient interpolation
            xp0 = orbital_time[0]  # Grid start time
            dx = orbital_time[1] - orbital_time[0]  # Grid spacing

            # Pre-interpolate all 6 orbital LTTs onto data time grid
            # This is done once rather than repeatedly in the loop
            ltt_interp = jax.vmap(
                lambda k: self._interp_uniform(t_arr, xp0, dx, orbital_ltt[k])
            )(jnp.arange(6, dtype=jnp.int32))  # Shape: (6,N)

            # Safe array indexing with bounds checking
            # Create mask for valid delay indices
            delay_mask = (delay_idx >= 0) & (delay_mask0[None, :, :])
            delay_idx_safe = jnp.where(delay_mask, delay_idx, 0)

            # Select orbital LTTs for all delay links
            selected = ltt_interp[delay_idx_safe]  # Shape: (3,T,L,N)

            # Sum LTTs along delay chain for each term
            total_lt = (selected * delay_mask[..., None]).sum(
                axis=2
            )  # Shape: (3,T,N)

            # Convert time delays to integer/fractional sample delays
            dts = (t_arr[None, None, :] - total_lt) * fs
            int_delays = jnp.ceil(dts).astype(jnp.int32) - 1
            fractions = 1.0 + int_delays - dts

            # Vectorized Lagrangian interpolation for all terms
            def interp_pt(p, t):
                """Interpolate data for one permutation and term."""
                bd = input_links[base_idx[p, t]]  # Base data for this term
                return jax.vmap(
                    lambda idl, frc: lagrangian_interpolation_real(
                        bd, idl, frc, A, E, deps
                    )
                )(int_delays[p, t], fractions[p, t])

            # Compute all channel terms for all permutations
            channels_pt = jax.vmap(
                lambda p: jax.vmap(lambda t: interp_pt(p, t))(
                    jnp.arange(signs.size, dtype=jnp.int32)
                )
            )(jnp.arange(3, dtype=jnp.int32))  # Shape: (3,T,N)

            # Sum terms with appropriate signs to get final channels
            return (channels_pt * signs[None, :, None]).sum(axis=1)  # (3,N)

        # Return JIT-compiled kernel
        # TODO: consider donate_argnums for large arrays
        return jax.jit(_kernel)

    def compute_tdi_channels(
        self,
        input_links: jnp.ndarray,  # (6,N)
        t_arr: jnp.ndarray,  # (N,)
        orbital_time: jnp.ndarray,  # (M,)
        orbital_ltt: jnp.ndarray,  # (6,M)
        tdi_type: str = "1st generation",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute TDI channels (X, Y, Z) from LISA interferometer data.

        Parameters
        ----------
        input_links : jnp.ndarray, shape (6, N)
            Interferometer data for all 6 LISA links. Links are ordered
            as [12, 23, 31, 13, 32, 21] corresponding to spacecraft pairs.
        t_arr : jnp.ndarray, shape (N,)
            Time array for the data points in seconds.
        orbital_time : jnp.ndarray, shape (M,)
            Time grid for orbital light travel time data. Should span
            at least the range of t_arr with sufficient padding.
        orbital_ltt : jnp.ndarray, shape (6, M)
            Orbital light travel times for all 6 links at orbital_time
            points. These account for the changing distances between
            spacecraft due to orbital motion.
        tdi_type : str, optional
            Type of TDI combination to use:
            - "1st generation": 8 terms, faster computation
            - "2nd generation": 16 terms, better noise cancellation
            Default is "1st generation".

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            TDI channel time series (X, Y, Z), each with shape (N,).
            These are the primary gravitational wave observables from LISA.

        Raises
        ------
        ValueError
            If tdi_type is not "1st generation" or "2nd generation".

        Notes
        -----
        For optimal performance, this method uses JIT-compiled kernels
        that are cached after the first call. Subsequent calls with
        similar array shapes will be much faster.

        Examples
        --------
        >>> processor = TDIProcessor(sampling_frequency=0.1)
        >>> X, Y, Z = processor.compute_tdi_channels(
        ...     input_links, t_arr, orbital_time, orbital_ltt
        ... )
        >>> # X, Y, Z are now the TDI observables
        """
        if tdi_type == "1st generation":
            channels = self._kernel_first(
                input_links, t_arr, orbital_time, orbital_ltt
            )
        elif tdi_type == "2nd generation":
            channels = self._kernel_second(
                input_links, t_arr, orbital_time, orbital_ltt
            )
        else:
            raise ValueError(
                "tdi_type must be '1st generation' or '2nd generation'"
            )

        return channels[0], channels[1], channels[2]
