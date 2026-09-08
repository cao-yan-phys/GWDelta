from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .array_backend import infer_backend_from_array
from .fastlisa import TDIResult
from .fd_response import (
    DEFAULT_LINKS,
    SECOND_GEN_X_COMBINATIONS,
    aet_from_xyz,
    cyclic_permutation,
)


def _as_numpy(value) -> np.ndarray:
    return np.asarray(infer_backend_from_array(value).asnumpy(value), dtype=float)


def _uniform_time_grid(t) -> tuple[np.ndarray, float]:
    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("t must be a one-dimensional array with at least two samples")
    step = np.diff(time)
    dt = float(np.median(step))
    if dt <= 0.0 or not np.allclose(step, dt, rtol=1.0e-10, atol=max(1.0e-12, abs(dt) * 1.0e-12)):
        raise ValueError("t must be strictly increasing and uniformly sampled")
    return time, dt


def compute_tdi2_ae(
    t,
    y_links,
    *,
    orbits: Any,
    t_buffer: float = 10_000.0,
) -> TDIResult:
    time, dt = _uniform_time_grid(t)
    links = _as_numpy(y_links)
    link_order = tuple(int(link) for link in orbits.LINKS)
    if link_order != DEFAULT_LINKS:
        raise ValueError(f"orbit link order must be {DEFAULT_LINKS}; got {link_order}")
    if links.shape != (len(link_order), len(time)):
        raise ValueError(
            f"y_links must have shape {(len(link_order), len(time))} in orbit link order; got {links.shape}"
        )
    if not np.all(np.isfinite(links)):
        raise ValueError("y_links must be finite")

    buffer_s = float(t_buffer)
    if not np.isfinite(buffer_s) or buffer_s <= 0.0:
        raise ValueError("t_buffer must be finite and positive")
    trim = int(buffer_s / dt)
    if trim < 1 or 2 * trim >= len(time):
        raise ValueError("t_buffer removes all samples")
    output_time = time[trim:-trim]

    orbit_time = _as_numpy(orbits.t_base)
    light_time = _as_numpy(orbits.ltt_base)
    if orbit_time.ndim != 1 or len(orbit_time) < 2 or np.any(np.diff(orbit_time) <= 0.0):
        raise ValueError("orbit time grid must be strictly increasing")
    if light_time.shape != (len(orbit_time), len(link_order)):
        raise ValueError("orbit light times have an unexpected shape")
    if np.any(~np.isfinite(light_time)) or np.any(light_time <= 0.0):
        raise ValueError("orbit light times must be finite and positive")

    light_time_splines = {
        link: CubicSpline(orbit_time, light_time[:, index], extrapolate=False)
        for index, link in enumerate(link_order)
    }
    link_splines = {
        link: CubicSpline(time, links[index], extrapolate=False)
        for index, link in enumerate(link_order)
    }
    xyz = []
    for permutation in range(3):
        channel = np.zeros_like(output_time)
        for term in SECOND_GEN_X_COMBINATIONS:
            delayed_time = output_time.copy()
            for delayed_link in term["links_for_delay"]:
                link = cyclic_permutation(int(delayed_link), permutation)
                delayed_time -= light_time_splines[link](delayed_time)
            if np.any(~np.isfinite(delayed_time)):
                raise ValueError("TDI delays leave the available orbit time range")
            base_link = cyclic_permutation(int(term["link"]), permutation)
            contribution = link_splines[base_link](delayed_time)
            if np.any(~np.isfinite(contribution)):
                raise ValueError("TDI delays leave the available link time range")
            channel += float(term["sign"]) * contribution
        xyz.append(channel)

    a_channel, e_channel, _t_channel = aet_from_xyz(*xyz)
    return TDIResult(
        t=output_time,
        channels={"A": a_channel, "E": e_channel},
        projections=None,
        response_model=None,
        metadata={
            "backend": "scipy_cubic_precomputed_tdi2",
            "input_kind": "precomputed_links",
            "tdi": "2nd generation",
            "tdi_requested": "2nd generation",
            "tdi_chan": "AE",
            "dt": dt,
            "t_buffer": buffer_s,
            "link_order": list(link_order),
            "tdi_start_ind": trim,
            "delay_evaluation": "recursive",
        },
    )
