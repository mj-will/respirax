import matplotlib.pyplot as plt
import numpy as np
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.utils.parallelbase import ParallelModuleBase
from lisatools.detector import EqualArmlengthOrbits

YRSID_SI = 31558149.763545603


# Parallel module base can help to facilitate customized gpu use.
class GBWave(ParallelModuleBase):
    def __init__(self, force_backend=None):
        super().__init__(force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        # get the t array
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
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


gb = GBWave(force_backend=None)

force_backend = None

T = 1.0  # years
t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

sampling_frequency = 0.1
dt = 1 / sampling_frequency

# order of the langrangian interpolation
order = 25

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "2nd generation"

index_lambda = 6
index_beta = 7

tdi_kwargs_esa = dict(
    order=order,
    tdi=tdi_gen,
    tdi_chan="XYZ",
)

gb_lisa_esa = ResponseWrapper(
    gb,
    T,
    dt,
    index_lambda,
    index_beta,
    t0=t0,
    flip_hx=False,  # set to True if waveform is h+ - ihx
    force_backend=force_backend,
    remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=True,  # False if using polar angle (theta)
    remove_garbage=True,  # removes the beginning of the signal that has bad information
    orbits=EqualArmlengthOrbits(),
    **tdi_kwargs_esa,
)

# define GB parameters
A = 1.084702251e-22
f = 2.35962078e-3
fdot = 1.47197271e-17
iota = 1.11820901
phi0 = 4.91128699
psi = 2.3290324

beta = 0.9805742971871619
lam = 5.22979888

chans = gb_lisa_esa(A, f, fdot, iota, phi0, psi, lam, beta)


fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
t_vec = np.arange(len(chans[0])) * dt / YRSID_SI
for channel, ax, label in zip(chans, axs, ["X", "Y", "Z"]):
    ax.plot(t_vec, channel)
    ax.set_ylabel(label)
axs[-1].set_xlabel("Time (yrs)")
plt.savefig("lisa_response_example_flr.png")
