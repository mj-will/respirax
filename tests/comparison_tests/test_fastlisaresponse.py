"""Test to ensure respirax and FLR response match"""

import numpy as np
import pytest

from respirax.orbital_utils import load_lisa_orbits
from respirax.response import LISAResponse
from respirax.utils import YRSID_SI

fastlisaresponse = pytest.importorskip(
    "fastlisaresponse",
    reason="fastlisaresponse not available",
)


@pytest.fixture
def intrinsic_parameters():
    return {
        "A": 1.084702251e-22,
        "f": 2.35962078e-3,
        "fdot": 1.47197271e-17,
        "iota": 1.11820901,
        "phi0": 4.91128699,
        "psi": 2.3290324,
    }


@pytest.fixture
def sky_parameters():
    return {
        "beta": 0.9805742971871619,
        "lam": 5.22979888,
    }


@pytest.fixture()
def flr_waveform_generator():
    from fastlisaresponse.utils.parallelbase import ParallelModuleBase

    class GBWave(ParallelModuleBase):
        def __init__(self, force_backend=None):
            super().__init__(force_backend=force_backend)

        @classmethod
        def supported_backends(cls):
            return [
                "fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()
            ]

        def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
            # get the t array
            # t = self.xp.arange(0.0, T * YRSID_SI, dt)

            # FIX for inconsistent handling of T_obs by FLR
            num_pts = int(T * YRSID_SI / dt)
            t = np.arange(num_pts) * dt

            cos2psi = self.xp.cos(2.0 * psi)
            sin2psi = self.xp.sin(2.0 * psi)
            cosiota = self.xp.cos(iota)

            fddot = 11.0 / 3.0 * fdot**2 / f

            # phi0 is phi(t = 0) not phi(t = t0)
            phase = (
                2
                * np.pi
                * (f * t + 1.0 / 2.0 * fdot * t**2 + 1.0 / 6.0 * fddot * t**3)
                - phi0
            )

            hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
            hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

            hp = hSp * cos2psi - hSc * sin2psi
            hc = hSp * sin2psi + hSc * cos2psi

            return hp + 1j * hc

    force_backend = None
    gb = GBWave(force_backend=force_backend)
    return gb


@pytest.fixture
def sampling_frequency():
    return 0.1


@pytest.fixture
def t_obs():
    return 0.25  # years


@pytest.fixture
def t0():
    return 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)


@pytest.fixture
def order():
    return 25


@pytest.fixture(params=["1st generation", "2nd generation"])
def tdi_generation(request):
    return request.param


@pytest.fixture
def flr_response(
    flr_waveform_generator,
    tdi_generation,
    sampling_frequency,
    t_obs,
    t0,
    order,
):
    from fastlisaresponse import ResponseWrapper as RefResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits

    dt = 1 / sampling_frequency

    # order of the langrangian interpolation
    index_lambda = 6
    index_beta = 7

    tdi_kwargs_esa = dict(
        order=order,
        tdi=tdi_generation,
        tdi_chan="XYZ",
    )

    gb_lisa_esa = RefResponseWrapper(
        flr_waveform_generator,
        t_obs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=False,  # set to True if waveform is h+ - ihx
        force_backend=None,
        remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=True,  # False if using polar angle (theta)
        remove_garbage=True,  # removes the beginning of the signal that has bad information
        orbits=EqualArmlengthOrbits(),
        **tdi_kwargs_esa,
    )
    return gb_lisa_esa


def test_projections(
    flr_waveform_generator,
    flr_response,
    intrinsic_parameters,
    sky_parameters,
    sampling_frequency,
    t_obs,
    t0,
    order,
):
    dt = 1 / sampling_frequency
    # define GB parameters
    waveform = flr_waveform_generator(
        intrinsic_parameters["A"],
        intrinsic_parameters["f"],
        intrinsic_parameters["fdot"],
        intrinsic_parameters["iota"],
        intrinsic_parameters["phi0"],
        intrinsic_parameters["psi"],
        T=t_obs,
        dt=dt,
    )

    flr_response.response_model.get_projections(
        waveform,
        sky_parameters["lam"],
        sky_parameters["beta"],
        t0=t0,
    )
    ref_projections = flr_response.response_model.y_gw

    orbits_file = flr_response.response_model.response_orbits.filename
    orbits_data = load_lisa_orbits(orbits_file)

    response = LISAResponse(
        sampling_frequency=sampling_frequency,
        num_pts=int(t_obs * sampling_frequency * YRSID_SI),
        order=order,
        orbits_data=orbits_data,
        t0=t0,
    )
    print("Calculating projections")
    projections = response.get_projections(
        waveform,
        sky_parameters["lam"],
        sky_parameters["beta"],
    )
    print("Checking projections")
    # FLR has shape (links, time)
    assert projections.shape == ref_projections.shape, (
        "Projections shape should match"
    )
    assert np.all(np.isfinite(projections)), "Projections should be finite"
    # Check values match reasonably
    for i in range(projections.shape[0]):
        print(f"Checking projection {i}")
        np.testing.assert_allclose(
            projections[i],
            ref_projections[i],
            atol=1e-32,
        )


def test_response(
    flr_waveform_generator,
    flr_response,
    intrinsic_parameters,
    sky_parameters,
    tdi_generation,
    sampling_frequency,
    t_obs,
    t0,
    order,
):
    dt = 1 / sampling_frequency
    # define GB parameters
    waveform = flr_waveform_generator(
        intrinsic_parameters["A"],
        intrinsic_parameters["f"],
        intrinsic_parameters["fdot"],
        intrinsic_parameters["iota"],
        intrinsic_parameters["phi0"],
        intrinsic_parameters["psi"],
        T=t_obs,
        dt=dt,
    )

    xyz_waveform_ref = flr_response(
        intrinsic_parameters["A"],
        intrinsic_parameters["f"],
        intrinsic_parameters["fdot"],
        intrinsic_parameters["iota"],
        intrinsic_parameters["phi0"],
        intrinsic_parameters["psi"],
        sky_parameters["lam"],
        sky_parameters["beta"],
    )
    ref_projections = flr_response.response_model.y_gw

    orbits_file = flr_response.response_model.response_orbits.filename
    orbits_data = load_lisa_orbits(orbits_file)

    response = LISAResponse(
        sampling_frequency=sampling_frequency,
        num_pts=int(t_obs * sampling_frequency * YRSID_SI),
        orbits_data=orbits_data,
        order=order,
        t0=t0,
    )

    interpolated_orbital_data = response.orbits_data
    # Check the orbital data matches
    orbits = flr_response.response_model.response_orbits
    assert np.allclose(orbits.t, interpolated_orbital_data["time"])
    assert np.isclose(orbits.dt, interpolated_orbital_data["dt"])

    # Assert light travel times match
    assert np.allclose(
        orbits.ltt, interpolated_orbital_data["light_travel_times"].T
    )

    # Assert x positions match
    for i in range(3):
        assert np.allclose(
            orbits.x[:, i, :], interpolated_orbital_data["positions"][i]
        )

    # lisatools returns (samples, link, xyz) for n whereas we have (link, xyz, samples)
    print(orbits.n.shape, interpolated_orbital_data["normal_vectors"].shape)
    assert np.allclose(
        orbits.n,
        interpolated_orbital_data["normal_vectors"].transpose(1, 0, 2),
    )
    print("Orbital data matches")
    xyz_waveform, projections = response.compute_response(
        waveform,
        sky_parameters["lam"],
        sky_parameters["beta"],
        tdi_type=tdi_generation,
        return_projections=True,
    )

    # FLR has shape (links, time)
    assert projections.shape == ref_projections.shape, (
        "Projections shape should match"
    )
    assert np.all(np.isfinite(projections)), "Projections should be finite"
    # Check values match reasonably
    np.testing.assert_allclose(
        projections,
        ref_projections,
        atol=1e-32,
    )
    print(len(xyz_waveform), len(xyz_waveform_ref))
    # Check waveform matches
    for i in range(len(xyz_waveform)):
        assert xyz_waveform[i].shape == xyz_waveform_ref[i].shape, (
            "Waveform shape should match"
        )
        np.testing.assert_allclose(
            xyz_waveform[i],
            xyz_waveform_ref[i],
            atol=1e-32,
        )
