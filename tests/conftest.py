import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    jax.config.update("jax_enable_x64", True)


@pytest.fixture
def orbits_file():
    from respirax.orbital_utils import get_orbit_file_path

    pytest.importorskip(
        "lisatools",
        reason="lisatools not available",
    )
    return get_orbit_file_path("equalarmlength")


@pytest.fixture
def orbits_data(orbits_file):
    from respirax.orbital_utils import load_lisa_orbits

    return load_lisa_orbits(orbits_file)
