import numpy as np
from scipy.interpolate import CubicSpline

from .fd_response import DEFAULT_LINKS, aet_from_xyz
from .recursive_tdi import _as_numpy, _validate_inputs


def compute_tdi2(t, y_links, *, orbits, tdi_chan="AE", t_buffer=2048., doppler=False):
    from .fastlisa import TDIResult

    if tdi_chan not in ("XYZ", "AET", "AE"):
        raise ValueError("tdi_chan must be XYZ, AET, or AE")
    if tuple(orbits.LINKS) != DEFAULT_LINKS:
        raise ValueError(f"Orbit link order must be {DEFAULT_LINKS}")
    orbit_time = _as_numpy(orbits.t_base)
    light_times = _as_numpy(orbits.ltt_base)
    if (orbit_time.ndim != 1 or len(orbit_time) < 2
            or not np.all(np.isfinite(orbit_time)) or np.any(np.diff(orbit_time) <= 0)
            or light_times.shape != (len(orbit_time), 6)
            or not np.all(np.isfinite(light_times)) or np.any(light_times <= 0)):
        raise ValueError("Invalid orbit light-time grid")
    raw_time = _as_numpy(t)
    if raw_time.ndim != 1 or len(raw_time) < 2:
        raise ValueError("Invalid time grid")
    if raw_time[0] < orbit_time[0] or raw_time[-1] > orbit_time[-1]:
        raise ValueError("Orbit coverage must contain the input time grid")
    sampled = CubicSpline(orbit_time, light_times, extrapolate=False)(raw_time).T
    time, dt, trim = _validate_inputs(raw_time, y_links, sampled, t_buffer)
    try:
        from pytdi import Data
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("The optional pytdi package is required") from exc
    links = _as_numpy(y_links)
    delays = {f"d_{link}": sampled[j] for j, link in enumerate(DEFAULT_LINKS)}
    measurements = {f"eta_{link}": links[j] for j, link in enumerate(DEFAULT_LINKS)}
    unit = "frequency" if doppler else "phase"
    data = Data({}, delays, 1 / dt)
    xyz = [compute_factorized_michelson(data, measurements, rot=j, generation=2,
                                      order=31, delay_order=5, unit=unit)
           for j in range(3)]
    values = xyz if tdi_chan == "XYZ" else aet_from_xyz(*xyz)
    names = "XYZ" if tdi_chan == "XYZ" else tdi_chan
    channels = {name: value[trim:-trim] for name, value in zip(names, values)}
    if any(not np.all(np.isfinite(value)) for value in channels.values()):
        raise ValueError("Nonfinite PyTDI output")
    return TDIResult(
        t=time[trim:-trim], channels=channels, projections=None, response_model=None,
        metadata={"backend": "pytdi", "tdi": "2nd generation", "tdi_chan": tdi_chan,
                  "dt": dt, "t_buffer": float(t_buffer), "doppler": bool(doppler),
                  "link_order": list(DEFAULT_LINKS), "input_kind": "precomputed_links",
                  "evaluation": "factorized", "signal_interpolation_order": 31,
                  "delay_interpolation_order": 5},
    )
