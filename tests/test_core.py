"""Core tests for respirax package functionality."""

import jax.numpy as jnp

from respirax import LISAResponse
from respirax.interpolation import (
    lagrangian_interpolation,
    prepare_interpolation_coefficients,
)
from respirax.tdi import TDIProcessor
from respirax.utils import get_basis_vecs, xi_projections

# Test constants
TOLERANCE = 1e-10
HIGH_PRECISION_TOLERANCE = 1e-14


class TestBasisVectors:
    """Test basis vector computations."""

    def test_basis_vector_shapes(self):
        """Test that basis vectors have correct shapes."""
        lam, beta = jnp.pi / 4, jnp.pi / 6
        u, v, k = get_basis_vecs(lam, beta)

        assert u.shape == (3,)
        assert v.shape == (3,)
        assert k.shape == (3,)

    def test_v_vector_normalization(self):
        """Test that v vector is normalized."""
        lam, beta = jnp.pi / 4, jnp.pi / 6
        u, v, k = get_basis_vecs(lam, beta)

        v_norm = jnp.sqrt(jnp.sum(v**2))
        # Use more relaxed tolerance due to float32 precision
        assert abs(v_norm - 1.0) < 1e-6

    def test_basis_vectors_different_sky_positions(self):
        """Test basis vectors at different sky positions."""
        positions = [
            (0.0, 0.0),  # Ecliptic pole
            (jnp.pi / 2, 0.0),  # Ecliptic equator
            (jnp.pi, jnp.pi / 4),  # General position
        ]

        for lam, beta in positions:
            u, v, k = get_basis_vecs(lam, beta)

            # v should always be normalized
            v_norm = jnp.sqrt(jnp.sum(v**2))
            assert abs(v_norm - 1.0) < 1e-6  # Relaxed tolerance

            # All vectors should have finite values
            assert jnp.all(jnp.isfinite(u))
            assert jnp.all(jnp.isfinite(v))
            assert jnp.all(jnp.isfinite(k))


class TestXiProjections:
    """Test xi projection calculations."""

    def test_xi_projections_simple_case(self):
        """Test xi projections with simple vectors."""
        u = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        n = jnp.array([1.0, 0.0, 0.0])  # aligned with u

        xi_p, xi_c = xi_projections(u, v, n)

        # When n is aligned with u: xi_p = 0.5, xi_c = 0
        assert abs(xi_p - 0.5) < TOLERANCE
        assert abs(xi_c - 0.0) < TOLERANCE

    def test_xi_projections_orthogonal_case(self):
        """Test xi projections when n is orthogonal to both u and v."""
        u = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        n = jnp.array([0.0, 0.0, 1.0])  # orthogonal to both

        xi_p, xi_c = xi_projections(u, v, n)

        # When n is orthogonal: xi_p = 0, xi_c = 0
        assert abs(xi_p) < TOLERANCE
        assert abs(xi_c) < TOLERANCE

    def test_xi_projections_finite_values(self):
        """Test that xi projections are always finite."""
        # Test with realistic basis vectors
        lam, beta = jnp.pi / 3, jnp.pi / 4
        u, v, k = get_basis_vecs(lam, beta)

        # Use k as a test direction
        xi_p, xi_c = xi_projections(u, v, k)

        assert jnp.isfinite(xi_p)
        assert jnp.isfinite(xi_c)


class TestInterpolation:
    """Test interpolation functionality."""

    def test_interpolation_coefficients_structure(self):
        """Test interpolation coefficient preparation."""
        order = 25
        A_coeffs, E_coeffs, deps = prepare_interpolation_coefficients(order)

        expected_h = (order + 1) // 2
        assert len(A_coeffs) == 1001  # Default num_A
        assert len(E_coeffs) == expected_h - 1
        assert abs(deps - 1.0 / 1000) < TOLERANCE

        # A coefficients should be positive
        assert jnp.all(A_coeffs > 0)

    def test_lagrangian_interpolation_polynomial(self):
        """Test interpolation accuracy with polynomial signals."""
        order = 25
        A_coeffs, E_coeffs, deps = prepare_interpolation_coefficients(order)

        # Create quadratic test signal
        dt = 0.1
        t = jnp.arange(1000) * dt
        a, b, c = 1.5, -0.8, 2.3
        real_part = a * t**2 + b * t + c
        imag_part = 0.5 * a * t**2 + 2 * b * t + 0.7 * c
        test_signal = real_part + 1j * imag_part

        # Test interpolation
        delay_index = 500
        fraction = 0.3

        hp_interp, hc_interp = lagrangian_interpolation(
            test_signal, delay_index, fraction, A_coeffs, E_coeffs, deps
        )

        # Compute exact values
        t_exact = (delay_index + 1 - fraction) * dt
        hp_exact = a * t_exact**2 + b * t_exact + c
        hc_exact = 0.5 * a * t_exact**2 + 2 * b * t_exact + 0.7 * c

        # High-order interpolation should be very accurate for polynomials
        # Use relaxed tolerance due to float32 precision
        assert abs(hp_interp - hp_exact) < 1e-3
        assert abs(hc_interp - hc_exact) < 1e-3

    def test_lagrangian_interpolation_sinusoid(self):
        """Test interpolation with sinusoidal signals."""
        order = 25
        A_coeffs, E_coeffs, deps = prepare_interpolation_coefficients(order)

        # Create smooth sinusoidal test signal
        dt = 0.1
        t = jnp.arange(1000) * dt
        freq = 0.005  # Low frequency for smooth interpolation
        hp_part = jnp.sin(2 * jnp.pi * freq * t)
        hc_part = jnp.cos(2 * jnp.pi * freq * t)
        test_signal = hp_part + 1j * hc_part

        delay_index = 400
        fraction = 0.7

        hp_interp, hc_interp = lagrangian_interpolation(
            test_signal, delay_index, fraction, A_coeffs, E_coeffs, deps
        )

        # For smooth functions, errors should be small
        expected_t = (delay_index + 1 - fraction) * dt
        expected_hp = jnp.sin(2 * jnp.pi * freq * expected_t)
        expected_hc = jnp.cos(2 * jnp.pi * freq * expected_t)

        assert abs(hp_interp - expected_hp) < 0.01
        assert abs(hc_interp - expected_hc) < 0.01


class TestTDI:
    """Test TDI functionality."""

    def test_tdi_processor_initialization(self):
        """Test TDI processor initialization."""
        processor = TDIProcessor(sampling_frequency=0.1)

        assert processor.sampling_frequency == 0.1
        assert processor.order == 25  # default

    def test_cyclic_permutation(self):
        """Test cyclic permutation logic."""
        processor = TDIProcessor(sampling_frequency=0.1)

        # Test basic permutations
        assert processor._cyclic_permutation(12, 0) == 12
        assert processor._cyclic_permutation(12, 1) == 23
        assert processor._cyclic_permutation(12, 2) == 31

        # Test wrap-around
        assert processor._cyclic_permutation(23, 1) == 31
        assert processor._cyclic_permutation(31, 1) == 12


class TestLISAResponse:
    """Test main LISAResponse class."""

    def test_lisa_response_initialization(self, orbits_data):
        """Test LISAResponse initialization."""
        lisa = LISAResponse(
            sampling_frequency=0.1,
            num_pts=100,
            order=15,
            orbits_data=orbits_data,
        )

        assert lisa.sampling_frequency == 0.1
        assert lisa.num_pts == 100
        assert lisa.order == 15

    def test_lisa_response_parameters(self, orbits_data):
        """Test parameter validation."""
        # Valid parameters
        lisa = LISAResponse(
            sampling_frequency=1.0,
            num_pts=1000,
            order=25,
            orbits_data=orbits_data,
        )
        assert lisa.sampling_frequency == 1.0

        # Test with different order
        lisa_custom = LISAResponse(
            sampling_frequency=0.5,
            num_pts=500,
            order=15,
            orbits_data=orbits_data,
        )
        assert lisa_custom.order == 15


class TestMathematicalConsistency:
    """Test mathematical consistency and properties."""

    def test_basis_vector_coordinate_system(self):
        """Test basis vector coordinate system properties."""
        lam, beta = jnp.pi / 6, jnp.pi / 4
        u, v, k = get_basis_vecs(lam, beta)

        # v should be normalized (this is enforced by design)
        v_norm = jnp.sqrt(jnp.sum(v**2))
        assert abs(v_norm - 1.0) < HIGH_PRECISION_TOLERANCE

        # k should point in the correct direction (toward source)
        # For positive beta, k_z component should be negative
        assert k[2] < 0  # This matches the CUDA implementation

    def test_interpolation_boundary_conditions(self):
        """Test interpolation at boundary conditions."""
        order = 25
        A_coeffs, E_coeffs, deps = prepare_interpolation_coefficients(order)

        # Create constant signal
        test_signal = jnp.ones(1000, dtype=complex)

        # Test interpolation with fraction = 0 (should return exact value)
        hp_interp, hc_interp = lagrangian_interpolation(
            test_signal, 500, 0.0, A_coeffs, E_coeffs, deps
        )

        # For constant signal, interpolation should return the constant
        # Note: lagrangian_interpolation returns complex values,
        # so we need to check real and imaginary parts
        assert abs(hp_interp - 1.0) < TOLERANCE
        assert abs(hc_interp - 0.0) < TOLERANCE  # Imaginary part should be 0

    def test_computation_stability(self):
        """Test numerical stability of computations."""
        # Test with extreme sky positions
        extreme_positions = [
            (0.0, jnp.pi / 2 - 1e-10),  # Near ecliptic pole
            (0.0, -jnp.pi / 2 + 1e-10),  # Near opposite pole
            (2 * jnp.pi - 1e-10, 0.0),  # Near longitude wrap-around
        ]

        for lam, beta in extreme_positions:
            u, v, k = get_basis_vecs(lam, beta)

            # All components should be finite
            assert jnp.all(jnp.isfinite(u))
            assert jnp.all(jnp.isfinite(v))
            assert jnp.all(jnp.isfinite(k))

            # v should still be normalized
            v_norm = jnp.sqrt(jnp.sum(v**2))
            assert abs(v_norm - 1.0) < 1e-6  # Relaxed tolerance
