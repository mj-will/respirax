"""Example usage of JAX LISA response functions."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from respirax import (
    LISAResponse,
    get_orbit_file_path,
    interpolate_orbital_data,
    load_lisa_orbits,
)
from respirax.utils import YRSID_SI

# Configure jax to use float64
jax.config.update("jax_enable_x64", True)


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


def main():
    T = 2.0  # years
    sampling_frequency = 0.1  # Hz
    dt = 1.0 / sampling_frequency

    # Get orbit file path, this requires lisaanalysistools
    # Alternatively, you can provide your own orbit file path here
    orbits_file = get_orbit_file_path(orbit_type="equalarmlength")
    # Load and interpolate LISA orbits
    orbit_data = load_lisa_orbits(orbits_file)

    # Interpolate on a regular grid for use with the response
    interp_data = interpolate_orbital_data(orbit_data, grid=True)

    # Define waveform parameters
    A = 1.084702251e-22
    f = 2.35962078e-3
    fdot = 1.47197271e-17
    iota = 1.11820901
    phi0 = 4.91128699
    psi = 2.3290324

    # Generate waveform
    h = waveform_generator(A, f, fdot, iota, phi0, psi, T=T, dt=dt)

    # Initialize LISA response
    response = LISAResponse(
        sampling_frequency=sampling_frequency,
        num_pts=int(T * sampling_frequency * YRSID_SI),
        order=25,
    )

    lam = 5.22979888  # radians
    beta = 0.9805742971871619  # radians

    # Compute TDI response
    channels = response(
        h,
        lam,
        beta,
        interp_data,
        t0=10000.0,
        tdi_type="2nd generation",
    )

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_vec = jnp.arange(len(channels[0])) * dt / YRSID_SI
    for channel, ax, label in zip(channels, axs, ["X", "Y", "Z"]):
        ax.plot(t_vec, channel)
        ax.set_ylabel(label)
    axs[-1].set_xlabel("Time (yrs)")
    plt.savefig("lisa_response_example.png")


if __name__ == "__main__":
    main()
