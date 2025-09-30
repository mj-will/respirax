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
    def __init__(self, sampling_frequency: float, order: int = 25):
        self.sampling_frequency = sampling_frequency
        self.dt = 1.0 / sampling_frequency
        self.order = order
        self.buffer_integer = order * 2 + 1

        # Precompute interpolation coefficients (fixed order)
        A, E, deps = prepare_interpolation_coefficients(order)
        self.A_coeffs, self.E_coeffs, self.deps = A, E, deps

        # Map: 12:0, 23:1, 31:2, 13:3, 32:4, 21:5
        self._link_code_to_idx = {12: 0, 23: 1, 31: 2, 13: 3, 32: 4, 21: 5}

        # Precompute term tensors and kernels
        self._terms_first = self._build_terms(FIRST_GEN_COMBS)
        self._terms_second = self._build_terms(
            FIRST_GEN_COMBS + SECOND_GEN_EXTRA
        )

        self._kernel_first = self._make_kernel(self._terms_first)
        self._kernel_second = self._make_kernel(self._terms_second)

    # ---------------- uniform-grid interpolation ----------------

    @staticmethod
    def _interp_uniform(x, xp0, dx, fp):
        """Linear interpolation for uniform grid xp = xp0 + dx*k."""
        k = (x - xp0) / dx
        i = jnp.floor(k).astype(jnp.int32)
        i = jnp.clip(i, 0, fp.size - 2)
        frac = k - i
        return fp[i] * (1.0 - frac) + fp[i + 1] * frac

    # ---------------- helpers for TDI setup ----------------

    @staticmethod
    def _cyclic_perm(links: jnp.ndarray, p: int):
        i = links // 10
        j = links % 10
        ip = ((i - 1 + p) % 3) + 1
        jp = ((j - 1 + p) % 3) + 1
        return 10 * ip + jp

    def _code_to_idx(self, links: jnp.ndarray):
        idx = jnp.full_like(links, -1)
        for code, k in self._link_code_to_idx.items():
            idx = jnp.where(links == code, k, idx)
        return idx

    def _pad_delays(self, delays_list: List[List[int]]):
        Lmax = max((len(d) for d in delays_list), default=0)
        arr = []
        for d in delays_list:
            arr.append(d + [-1] * (Lmax - len(d)))
        return jnp.array(arr, dtype=jnp.int32), jnp.array(
            [[v >= 0 for v in row] for row in arr], dtype=bool
        )

    def _build_terms(self, combo_list: List[Dict]):
        base = jnp.array(
            [c["link"] for c in combo_list], dtype=jnp.int32
        )  # (T,)
        signs = jnp.array(
            [c["sign"] for c in combo_list], dtype=jnp.int32
        )  # (T,)
        delays_raw = [c["links_for_delay"] for c in combo_list]
        delay_links, delay_mask = self._pad_delays(delays_raw)  # (T,L), (T,L)

        T, L = delay_links.shape
        base_perm = jnp.stack(
            [self._cyclic_perm(base, p) for p in (0, 1, 2)], axis=0
        )  # (3,T)
        delay_perm = jnp.stack(
            [self._cyclic_perm(delay_links, p) for p in (0, 1, 2)], 0
        )  # (3,T,L)

        base_idx = self._code_to_idx(base_perm)  # (3,T)
        delay_idx = self._code_to_idx(delay_perm)  # (3,T,L)

        return {
            "signs": signs,
            "base_idx": base_idx,
            "delay_idx": delay_idx,
            "delay_mask": delay_mask,
        }

    def _compute_delay_int_frac(self, t_arr, total_lt):
        dts = (t_arr[None, None, :] - total_lt) * self.sampling_frequency
        int_delays = jnp.ceil(dts).astype(jnp.int32) - 1
        fractions = 1.0 + int_delays - dts
        return int_delays, fractions

    def _lagrange_apply(self, base_data, int_delays, fractions):
        def interp_one(int_d, frac):
            return lagrangian_interpolation_real(
                base_data, int_d, frac, self.A_coeffs, self.E_coeffs, self.deps
            )

        return jax.vmap(interp_one)(int_delays, fractions)

    # ---------------- kernel factory ----------------

    def _make_kernel(self, terms):
        signs = terms["signs"]
        base_idx = terms["base_idx"]
        delay_idx = terms["delay_idx"]
        delay_mask0 = terms["delay_mask"]

        A, E, deps = self.A_coeffs, self.E_coeffs, self.deps
        fs = self.sampling_frequency

        def _kernel(input_links, t_arr, orbital_time, orbital_ltt):
            """
            input_links: (6,N), t_arr:(N,), orbital_time:(M,), orbital_ltt:(6,M)
            returns (3,N)
            """
            xp0 = orbital_time[0]
            dx = orbital_time[1] - orbital_time[0]

            # Pre-interpolate all 6 LTTs onto t_arr once → (6,N)
            ltt_interp = jax.vmap(
                lambda k: self._interp_uniform(t_arr, xp0, dx, orbital_ltt[k])
            )(jnp.arange(6, dtype=jnp.int32))

            # Safe gather for delays
            delay_mask = (delay_idx >= 0) & (delay_mask0[None, :, :])
            delay_idx_safe = jnp.where(delay_mask, delay_idx, 0)
            selected = ltt_interp[delay_idx_safe]  # (3,T,L,N)
            total_lt = (selected * delay_mask[..., None]).sum(
                axis=2
            )  # (3,T,N)

            # Integer/fractional delays
            dts = (t_arr[None, None, :] - total_lt) * fs
            int_delays = jnp.ceil(dts).astype(jnp.int32) - 1
            fractions = 1.0 + int_delays - dts

            # Interpolate base data
            def interp_pt(p, t):
                bd = input_links[base_idx[p, t]]
                return jax.vmap(
                    lambda idl, frc: lagrangian_interpolation_real(
                        bd, idl, frc, A, E, deps
                    )
                )(int_delays[p, t], fractions[p, t])

            channels_pt = jax.vmap(
                lambda p: jax.vmap(lambda t: interp_pt(p, t))(
                    jnp.arange(signs.size, dtype=jnp.int32)
                )
            )(jnp.arange(3, dtype=jnp.int32))  # (3,T,N)

            return (channels_pt * signs[None, :, None]).sum(axis=1)  # (3,N)

        return jax.jit(_kernel, donate_argnums=(0,))

    # ---------------- public API ----------------

    def compute_tdi_channels(
        self,
        input_links: jnp.ndarray,  # (6,N)
        t_arr: jnp.ndarray,  # (N,)
        orbital_time: jnp.ndarray,  # (M,)
        orbital_ltt: jnp.ndarray,  # (6,M)
        tdi_type: str = "1st generation",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
