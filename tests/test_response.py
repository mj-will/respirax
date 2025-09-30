import jax.numpy as jnp
import numpy as np

from respirax import LISAResponse
from respirax.orbital_utils import interpolate_orbital_data, load_lisa_orbits
from respirax.utils import YRSID_SI


def waveform_generator(A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
    """Galactic binary waveform based on tutorial example."""
    # get the t array
    t = jnp.arange(0.0, T * YRSID_SI, dt)
    cos2psi = jnp.cos(2.0 * psi)
    sin2psi = jnp.sin(2.0 * psi)
    cosiota = jnp.cos(iota)

    fddot = 11.0 / 3.0 * fdot**2 / f

    # phi0 is phi(t = 0) not phi(t = t0)
    phase = (
        2
        * jnp.pi
        * (f * t + 1.0 / 2.0 * fdot * t**2 + 1.0 / 6.0 * fddot * t**3)
        - phi0
    )

    hSp = -jnp.cos(phase) * A * (1.0 + cosiota * cosiota)
    hSc = -jnp.sin(phase) * 2.0 * A * cosiota

    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi

    return hp + 1j * hc


def test_respirax_response(orbits_file):
    # Generate waveform with respirax for comparison
    T = 0.1  # years
    t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

    sampling_frequency = 0.1
    dt = 1 / sampling_frequency

    # order of the langrangian interpolation
    order = 5

    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"

    A = 1.084702251e-22
    f = 2.35962078e-3
    fdot = 1.47197271e-17
    iota = 1.11820901
    phi0 = 4.91128699
    psi = 2.3290324

    beta = 0.9805742971871619
    lam = 5.22979888

    response = LISAResponse(
        sampling_frequency=sampling_frequency,
        num_pts=int(T * sampling_frequency * YRSID_SI),
        order=order,
    )

    print("Generating waveform...")
    waveform = waveform_generator(A, f, fdot, iota, phi0, psi, T=T, dt=dt)

    orbital_data = load_lisa_orbits(orbits_file)

    interpolated_orbital_data = interpolate_orbital_data(
        orbital_data,
        grid=True,
    )

    # Get projections
    response, projections = response.compute_response(
        waveform,
        lam,
        beta,
        interpolated_orbital_data,
        t0=t0,
        tdi_type=tdi_gen,
        return_projections=True,
    )
    assert np.all(np.isfinite(projections)), "Projections should be finite"
