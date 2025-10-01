"""JAX-compatible orbital data loader and utilities."""

from pathlib import Path
from typing import Dict, Optional

import h5py
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline


def get_orbit_file_path(
    orbit_type: str = "equalarmlength", fallback_path: Optional[str] = None
) -> Path:
    """Get path to LISA orbit file, trying lisaanalysistools first.

    Parameters
    ----------
    orbit_type : str, optional
        Type of orbit file ('equalarmlength', 'esa', etc.), by default "equalarmlength"
    fallback_path : str, optional
        Manual path to use if lisaanalysistools not available, by default None

    Returns
    -------
    Path
        Path to orbit file

    Raises
    ------
    ImportError
        If lisaanalysistools not available and no fallback provided
    FileNotFoundError
        If orbit file cannot be found
    """
    # Try to use lisaanalysistools if available
    try:
        from lisatools.detector import EqualArmlengthOrbits, ESAOrbits

        if orbit_type.lower() == "equalarmlength":
            orbits = EqualArmlengthOrbits()
            return Path(orbits.filename)
        elif orbit_type.lower() == "esa":
            orbits = ESAOrbits()
            return Path(orbits.filename)
        else:
            # Try to construct filename
            filename = f"{orbit_type}-orbits.h5"
            orbits = (
                EqualArmlengthOrbits()
            )  # Use as base to get path resolution
            orbits.filename = filename  # This will download if needed
            return Path(orbits.filename)

    except ImportError:
        if fallback_path is None:
            raise ImportError(
                "lisaanalysistools not available and no fallback_path provided. "
                "Install lisaanalysistools with 'pip install lisaanalysistools' "
                "or provide a manual path to an orbit file."
            )

        fallback = Path(fallback_path)
        if not fallback.exists():
            raise FileNotFoundError(
                f"Fallback orbit file not found: {fallback}"
            )
        return fallback


def load_lisa_orbits(
    orbit_file: str | Path,
) -> Dict:
    """Load LISA orbital data in JAX-compatible format.

    Parameters
    ----------
    orbit_file : str or Path
        Path to orbit file

    Returns
    -------
    Dict
        Dictionary with orbital data in JAX arrays containing:

        - time : jnp.ndarray
            Time array
        - positions : list of jnp.ndarray
            Spacecraft positions for each spacecraft
        - light_travel_times : jnp.ndarray
            Light travel times for each link, shape (6, N)
        - normal_vectors : jnp.ndarray
            Normal vectors for each link, shape (6, 3, N)
        - velocities : jnp.ndarray
            Spacecraft velocities
        - dt : float
            Time step
        - links : list
            LISA link convention [12, 23, 31, 13, 32, 21]
        - source_file : Path
            Source file path
    """
    if orbit_file is None:
        raise ValueError("orbit_file must be provided")
    orbit_file = Path(orbit_file)
    if not orbit_file.exists():
        raise FileNotFoundError(f"Orbit file not found: {orbit_file}")

    # Load orbital data
    with h5py.File(orbit_file, "r") as f:
        # Get metadata
        dt_base = f.attrs["dt"]
        size_base = f.attrs["size"]

        # Create time array
        t_base = np.arange(size_base) * dt_base

        # Spacecraft positions [time, spacecraft, xyz]
        x_base = f["tcb"]["x"][:]

        # Light travel times [time, link]
        ltt_base = f["tcb"]["ltt"][:]

        # Velocities
        v_base = f["tcb"]["v"][:]

        # Normal vectors [time, link, xyz]
        n_base = f["tcb"]["n"][:]

    # Convert to JAX format
    time_array = jnp.array(t_base[:])

    # Organize spacecraft positions
    positions = [jnp.array(x_base[:, i, :]) for i in range(3)]

    # Transpose light travel times to [link, time] format
    light_travel_times = jnp.array(ltt_base).T

    velocities = jnp.array(v_base)

    # Reshape normal vectors to [link, xyz, time]
    normal_vectors = jnp.array(n_base).transpose(1, 2, 0)

    return {
        "time": time_array,
        "positions": positions,
        "light_travel_times": light_travel_times,
        "normal_vectors": normal_vectors,
        "velocities": velocities,
        "dt": dt_base,
        "links": [12, 23, 31, 13, 32, 21],  # LISA link convention
        "source_file": orbit_file,
    }


def interpolate_orbital_data(
    data: Dict,
    dt: float | None = None,
    times: jnp.ndarray | None = None,
    grid: bool = False,
) -> Dict:
    """Interpolate orbital data to a new time step using cubic splines.

    Parameters
    ----------
    data : Dict
        Original orbital data dictionary
    dt : float, optional
        Desired time step in seconds, by default None
    times : jnp.ndarray, optional
        Specific time array to interpolate to, by default None
    grid : bool, optional
        Whether to use a regular grid (dt=600s), by default False

    Returns
    -------
    Dict
        New orbital data dictionary with interpolated values
    """
    original_time = data["time"]

    t_obs = None

    if grid:
        dt = 600

    if times is not None:
        if times[0] < original_time[1] or times[-1] > original_time[-1]:
            raise ValueError("Requested times are outside data")
        dt = abs(times[1] - times[1])
    elif dt is None:
        times = original_time.copy()
    else:
        t_obs = original_time[-1] - original_time[0]
        n_obs = int(t_obs / dt)
        times = jnp.arange(n_obs) * dt
        if times[-1] < original_time[-1]:
            times = jnp.concatenate([times, original_time[-1:]])

    # Interpolate positions for each spacecraft
    new_positions = []
    for pos in data["positions"]:
        cs_x = CubicSpline(original_time, pos[:, 0])
        cs_y = CubicSpline(original_time, pos[:, 1])
        cs_z = CubicSpline(original_time, pos[:, 2])
        new_pos = jnp.stack([cs_x(times), cs_y(times), cs_z(times)], axis=1)
        new_positions.append(new_pos)

    # Interpolate light travel times for each link
    ltt = data["light_travel_times"]
    new_ltt = []
    for i in range(ltt.shape[0]):
        cs_ltt = CubicSpline(original_time, ltt[i, :])
        new_ltt.append(cs_ltt(times))
    new_light_travel_times = jnp.array(new_ltt)

    # Interpolate normal vectors for each link and component
    n_vecs = data["normal_vectors"]
    new_n_vecs = []
    for i in range(n_vecs.shape[0]):  # links
        new_n_link = []
        for j in range(n_vecs.shape[1]):  # xyz components
            cs_n = CubicSpline(original_time, n_vecs[i, j, :])
            new_n_link.append(cs_n(times))
        new_n_vecs.append(jnp.array(new_n_link).T)  # shape (time, xyz)
    new_normal_vectors = jnp.array(new_n_vecs)  # shape (links, time, xyz)

    return {
        "time": times,
        "positions": new_positions,
        "light_travel_times": new_light_travel_times,
        "normal_vectors": new_normal_vectors,
        "dt": dt,
        "links": data["links"],
        "source_file": data.get("source_file", None),
    }
