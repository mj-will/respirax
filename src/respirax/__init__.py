"""JAX implementation of LISA response functions.

This package provides a JAX-based implementation of the LISA response functions,
including time-domain interpolation and TDI (Time-Delay Interferometry) computations.

This package is based on fastlisaresponse, if you use this code please cite
that package.
"""

from importlib.metadata import PackageNotFoundError, version

from .orbital_utils import (
    get_orbit_file_path,
    interpolate_orbital_data,
    load_lisa_orbits,
)
from .response import LISAResponse
from .utils import xyz_to_aet

try:
    __version__ = version("respirax")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "LISAResponse",
    "load_lisa_orbits",
    "interpolate_orbital_data",
    "get_orbit_file_path",
    "xyz_to_aet",
]
