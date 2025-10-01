# Respirax

A JAX-based implementation of the LISA response functions, converted from the
implementation in `fastlisaresponse`.

## Installation

`respirax` can be installed from source using:

```
pip install git+https://github.com/mj-will/respirax.git
```

## Basic usage

### Orbits

`respirax` does not currently include orbital data for lisa. You either need to
provide the files yourself or install `lisatools`.

```python
# If you have lisatools installed
orbits_file = get_orbit_file_path(orbit_type="equalarmlength")
orbits_data = load_lisa_orbits(orbits_file)
```

### Calculating the response

```python
import jax
from respirax import LISAResponse
from respirax.utils import YRSID_SI

# Recommend using
jax.config.update("jax_enable_x64", True)

t_obs = 1.0 # 1 year
sampling_frequency = 1 # Hz
num_pts = int(t_obs * sampling_frequency * YRSID_SI)

response  = LISAResponse(
    sampling_frequency=sampling_frequency,
    num_pts=num_pts,
    order=25,    # Order of the Lagragian interpolation
    orbits_data=orbits_data,
    t0=10000.0,    # Start time buffer in seconds
)

# Generate a complex waveform as h_+ + i*h_x
waveform = ...

# Sky location
lam = 0.56
beta = 0.12

# Apply the response
X, Y, T = response(waveform, lam, beta, tdi_type="1st generation")

# Alternatively, return the A, E, T channels by specifying `tdi_channels`
A, E, T = response(waveform, lam, beta, tdi_type="1st generation", tdi_channels="AET")
```

## Citation

Please cite this repo and the original `fastlisareponse` code following the instructions for that code.
