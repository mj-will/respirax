"""Tests for orbital data loading and interpolation compatibility."""

import numpy as np
import pytest

from respirax.orbital_utils import interpolate_orbital_data, load_lisa_orbits

lisatools = pytest.importorskip(
    "lisatools",
    reason="lisatools not available",
)


@pytest.mark.comparison
def test_orbital_data():
    """Test that our orbital data loading matches fastlisaresponse approach."""
    from lisatools.detector import EqualArmlengthOrbits

    orbits = EqualArmlengthOrbits()

    # Load with respirax
    respirax_orbits = load_lisa_orbits(orbits.filename)

    # Import EqualArmlengthOrbits from lisatools

    assert respirax_orbits["dt"] == orbits.dt_base, "Time step mismatch"
    assert np.allclose(respirax_orbits["time"], orbits.t_base)
    assert np.allclose(respirax_orbits["positions"][0], orbits.x_base[:, 0, :])


@pytest.mark.comparison
def test_orbital_data_interpolation():
    """Test interpolation of orbital data matches fastlisaresponse approach."""
    from lisatools.detector import EqualArmlengthOrbits

    orbits = EqualArmlengthOrbits()

    # Load with respirax
    respirax_orbits = load_lisa_orbits(orbits.filename)

    orbits.configure(linear_interp_setup=True)

    respirax_orbits_interp = interpolate_orbital_data(
        respirax_orbits, grid=True
    )

    assert np.allclose(orbits.t, respirax_orbits_interp["time"])
    assert np.isclose(orbits.dt, respirax_orbits_interp["dt"])

    # Assert light travel times match
    assert np.allclose(
        orbits.ltt, respirax_orbits_interp["light_travel_times"].T
    )

    # Assert x positions match
    for i in range(3):
        assert np.allclose(
            orbits.x[:, i, :], respirax_orbits_interp["positions"][i]
        )

    # lisatools returns (samples, link, xyz) for n whereas we have (link, xyz, samples)
    print(orbits.n.shape, respirax_orbits_interp["normal_vectors"].shape)
    assert np.allclose(
        orbits.n, respirax_orbits_interp["normal_vectors"].transpose(1, 0, 2)
    )
