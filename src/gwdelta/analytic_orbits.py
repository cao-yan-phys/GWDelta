"""Analytic LISA-like orbit helpers in the standard TDI convention.

Triangle-Simulator Taiji orbit files use the opposite spacecraft winding from
the standard LISA/TDI convention used by the analytic response formulas.  This
module keeps that relabeling explicit and provides the analytic equal-arm orbit
used for static response checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .fd_response import C_SI, DEFAULT_LINKS, TAIJI_ARM_M
from .orbits import LINEAR_INTERP_TIMESTEP, build_orbit_arrays, SampledOrbits


RAW_TO_STANDARD_SC = (1, 0, 2)
RAW_TO_STANDARD_LTT = (5, 3, 4, 1, 2, 0)
SIDEREAL_YEAR_S = 31_558_149.763545603


@dataclass(frozen=True)
class StaticOrbitMatch:
    """Static equal-arm orbit matched to a reference triangle."""

    positions_m: np.ndarray
    center_m: np.ndarray
    arm_m: float
    eta_rad: float
    phi0_rad: float
    fit_rms_m: float
    reference_time_s: float


@dataclass(frozen=True)
class DynamicOrbitMatch:
    """Dynamic equal-arm orbit matched to a reference triangle."""

    positions_m_at_reference: np.ndarray
    center_m_at_reference: np.ndarray
    arm_m: float
    eta_rad_at_reference: float
    phi0_rad: float
    fit_rms_m: float
    reference_time_s: float
    center_radius_m: float
    center_z_m: float
    center_period_s: float
    orbit_dt_s: float


def standard_detector_frame_positions(arm_m: float = TAIJI_ARM_M) -> np.ndarray:
    """Return detector-frame spacecraft positions in standard labels."""

    indices = np.asarray([1.0, 2.0, 3.0])
    sigma = np.pi / 4.0 + indices * 2.0 * np.pi / 3.0
    return float(arm_m) / np.sqrt(3.0) * np.column_stack(
        [np.cos(2.0 * sigma), np.sin(2.0 * sigma), np.zeros(3)]
    )


def analytic_ssb_rotation_matrix(eta: float | np.ndarray, phi0: float) -> np.ndarray:
    """Return the analytic rotation matrix mapping detector to SSB coordinates."""

    eta_arr = np.asarray(eta, dtype=float)
    phi0 = float(phi0)
    s_phi = np.sin(phi0)
    c_phi = np.cos(phi0)
    s_2 = np.sin(2.0 * eta_arr - phi0)
    c_2 = np.cos(2.0 * eta_arr - phi0)
    s_eta = np.sin(eta_arr)
    c_eta = np.cos(eta_arr)
    s_em = np.sin(eta_arr - phi0)
    c_em = np.cos(eta_arr - phi0)

    r = np.empty(eta_arr.shape + (3, 3), dtype=float)
    sqrt3 = np.sqrt(3.0)
    r[..., 0, 0] = sqrt3 * (3.0 * s_phi + s_2) - c_2 + 3.0 * c_phi
    r[..., 0, 1] = -s_2 - sqrt3 * c_2 - 3.0 * s_phi + 3.0 * sqrt3 * c_phi
    r[..., 0, 2] = -4.0 * sqrt3 * c_eta
    r[..., 1, 0] = -s_2 - sqrt3 * c_2 + 3.0 * s_phi - 3.0 * sqrt3 * c_phi
    r[..., 1, 1] = sqrt3 * (3.0 * s_phi - s_2) + c_2 + 3.0 * c_phi
    r[..., 1, 2] = -4.0 * sqrt3 * s_eta
    r[..., 2, 0] = 2.0 * sqrt3 * c_em - 6.0 * s_em
    r[..., 2, 1] = 2.0 * sqrt3 * s_em + 6.0 * c_em
    r[..., 2, 2] = 4.0
    return r / 8.0


def analytic_relative_positions(
    eta: float,
    phi0: float,
    *,
    arm_m: float = TAIJI_ARM_M,
) -> np.ndarray:
    """Return analytic equal-arm relative positions in SSB coordinates."""

    detector_positions = standard_detector_frame_positions(arm_m)
    rotation = analytic_ssb_rotation_matrix(float(eta), float(phi0))
    return np.einsum("ij,kj->ki", rotation, detector_positions)


def analytic_relative_position_series(
    eta: np.ndarray,
    phi0: float,
    *,
    arm_m: float = TAIJI_ARM_M,
) -> np.ndarray:
    """Return analytic equal-arm relative positions for a series of SSB phases."""

    detector_positions = standard_detector_frame_positions(arm_m)
    rotation = analytic_ssb_rotation_matrix(np.asarray(eta, dtype=float), float(phi0))
    return np.einsum("nij,kj->nki", rotation, detector_positions)


def relabel_raw_spacecraft_to_standard(x_raw: np.ndarray) -> np.ndarray:
    """Swap raw SC1/SC2 into the standard LISA/TDI convention."""

    x = np.asarray(x_raw, dtype=float)
    return x[..., RAW_TO_STANDARD_SC, :]


def relabel_raw_ltt_to_standard(ltt_raw: np.ndarray) -> np.ndarray:
    """Remap light travel times in link order [12, 23, 31, 13, 32, 21]."""

    ltt = np.asarray(ltt_raw, dtype=float)
    return ltt[..., RAW_TO_STANDARD_LTT]


def make_standard_convention_orbits(raw_orbits, *, force_backend: str | None = None) -> SampledOrbits:
    """Return a relabeled copy of a raw Taiji/LISA orbit object."""

    x = relabel_raw_spacecraft_to_standard(raw_orbits.x_base)
    v = relabel_raw_spacecraft_to_standard(raw_orbits.v_base)
    ltt = relabel_raw_ltt_to_standard(raw_orbits.ltt_base)
    return SampledOrbits(
        raw_orbits.t_base,
        x,
        v=v,
        ltt=ltt,
        links=tuple(DEFAULT_LINKS),
        armlength=float(raw_orbits.armlength),
        force_backend=force_backend,
    )


def median_arm_length(positions_m: np.ndarray) -> float:
    """Median arm length for positions in link order [12, 23, 31, 13, 32, 21]."""

    positions = np.asarray(positions_m, dtype=float)
    receivers = np.asarray([int(str(link)[0]) - 1 for link in DEFAULT_LINKS])
    emitters = np.asarray([int(str(link)[1]) - 1 for link in DEFAULT_LINKS])
    return float(np.median(np.linalg.norm(positions[receivers] - positions[emitters], axis=-1)))


def fit_analytic_phi0_to_reference(
    reference_positions_m: np.ndarray,
    *,
    eta: float | None = None,
    arm_m: float | None = None,
) -> tuple[float, float]:
    """Fit the analytic phase constant to a standard-labeled reference triangle."""

    positions = np.asarray(reference_positions_m, dtype=float)
    if positions.shape != (3, 3):
        raise ValueError("reference_positions_m must have shape (3, 3)")
    center = positions.mean(axis=0)
    rel = positions - center
    eta_val = float(np.arctan2(center[1], center[0]) if eta is None else eta)
    arm_val = median_arm_length(positions) if arm_m is None else float(arm_m)

    def objective(phi0: float) -> float:
        model = analytic_relative_positions(eta_val, phi0, arm_m=arm_val)
        diff = model - rel
        return float(np.mean(np.sum(diff * diff, axis=-1)))

    grid = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
    values = np.asarray([objective(item) for item in grid])
    best = float(grid[int(np.argmin(values))])
    width = 2.0 * np.pi / len(grid)
    result = minimize_scalar(objective, bounds=(best - width, best + width), method="bounded")
    phi0 = float(result.x % (2.0 * np.pi))
    rms = float(np.sqrt(objective(phi0)))
    return phi0, rms


def make_static_equal_arm_orbits_from_reference(
    reference_positions_m: np.ndarray,
    *,
    duration_s: float,
    reference_time_s: float,
    eta: float | None = None,
    arm_m: float | None = None,
    center_at_reference: bool = True,
    force_backend: str | None = None,
) -> tuple[SampledOrbits, StaticOrbitMatch]:
    """Build a static equal-arm orbit matched to a standard-labeled triangle."""

    positions = np.asarray(reference_positions_m, dtype=float)
    center = positions.mean(axis=0)
    eta_val = float(np.arctan2(center[1], center[0]) if eta is None else eta)
    arm_val = median_arm_length(positions) if arm_m is None else float(arm_m)
    phi0, fit_rms = fit_analytic_phi0_to_reference(positions, eta=eta_val, arm_m=arm_val)
    static_center = center if center_at_reference else np.zeros(3, dtype=float)
    static_positions = static_center + analytic_relative_positions(eta_val, phi0, arm_m=arm_val)
    t_orbit = np.asarray([0.0, float(duration_s)], dtype=float)
    x_orbit = np.repeat(static_positions[np.newaxis, :, :], 2, axis=0)
    orbits = SampledOrbits(
        t_orbit,
        x_orbit,
        links=tuple(DEFAULT_LINKS),
        armlength=arm_val,
        force_backend=force_backend,
    )
    match = StaticOrbitMatch(
        positions_m=static_positions,
        center_m=static_center,
        arm_m=arm_val,
        eta_rad=eta_val,
        phi0_rad=phi0,
        fit_rms_m=fit_rms,
        reference_time_s=float(reference_time_s),
    )
    return orbits, match


def make_dynamic_equal_arm_orbits_from_reference(
    reference_positions_m: np.ndarray,
    *,
    duration_s: float,
    reference_time_s: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    center_period_s: float = SIDEREAL_YEAR_S,
    eta: float | None = None,
    arm_m: float | None = None,
    center_radius_m: float | None = None,
    center_z_m: float | None = None,
    force_backend: str | None = None,
) -> tuple[SampledOrbits, DynamicOrbitMatch]:
    """Build a dynamic analytic equal-arm orbit matched to a reference triangle."""

    positions = np.asarray(reference_positions_m, dtype=float)
    if positions.shape != (3, 3):
        raise ValueError("reference_positions_m must have shape (3, 3)")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if orbit_dt <= 0.0:
        raise ValueError("orbit_dt must be positive")
    if center_period_s <= 0.0:
        raise ValueError("center_period_s must be positive")

    center = positions.mean(axis=0)
    eta_ref = float(np.arctan2(center[1], center[0]) if eta is None else eta)
    arm_val = median_arm_length(positions) if arm_m is None else float(arm_m)
    phi0, fit_rms = fit_analytic_phi0_to_reference(positions, eta=eta_ref, arm_m=arm_val)
    radius = float(np.linalg.norm(center[:2]) if center_radius_m is None else center_radius_m)
    center_z = float(center[2] if center_z_m is None else center_z_m)

    n = int(np.ceil(float(duration_s) / float(orbit_dt))) + 1
    t_orbit = np.arange(n, dtype=float) * float(orbit_dt)
    if t_orbit[-1] < duration_s:
        t_orbit = np.concatenate([t_orbit, np.asarray([float(duration_s)])])
    else:
        t_orbit[-1] = float(duration_s)

    eta_series = eta_ref + 2.0 * np.pi * (t_orbit - float(reference_time_s)) / float(center_period_s)
    center_series = np.column_stack(
        [
            radius * np.cos(eta_series),
            radius * np.sin(eta_series),
            np.full_like(eta_series, center_z),
        ]
    )
    relative = analytic_relative_position_series(eta_series, phi0, arm_m=arm_val)
    x_orbit = center_series[:, None, :] + relative
    arrays = build_orbit_arrays(t_orbit, x_orbit, links=tuple(DEFAULT_LINKS), armlength=arm_val)
    orbits = SampledOrbits(
        arrays.t,
        arrays.x,
        v=arrays.v,
        ltt=arrays.ltt,
        links=arrays.links,
        armlength=arrays.armlength,
        force_backend=force_backend,
    )
    ref_relative = analytic_relative_positions(eta_ref, phi0, arm_m=arm_val)
    match = DynamicOrbitMatch(
        positions_m_at_reference=np.asarray([radius * np.cos(eta_ref), radius * np.sin(eta_ref), center_z]) + ref_relative,
        center_m_at_reference=np.asarray([radius * np.cos(eta_ref), radius * np.sin(eta_ref), center_z]),
        arm_m=arm_val,
        eta_rad_at_reference=eta_ref,
        phi0_rad=phi0,
        fit_rms_m=fit_rms,
        reference_time_s=float(reference_time_s),
        center_radius_m=radius,
        center_z_m=center_z,
        center_period_s=float(center_period_s),
        orbit_dt_s=float(orbit_dt),
    )
    return orbits, match


__all__ = [
    "DynamicOrbitMatch",
    "StaticOrbitMatch",
    "RAW_TO_STANDARD_LTT",
    "RAW_TO_STANDARD_SC",
    "SIDEREAL_YEAR_S",
    "fit_analytic_phi0_to_reference",
    "make_dynamic_equal_arm_orbits_from_reference",
    "make_standard_convention_orbits",
    "make_static_equal_arm_orbits_from_reference",
    "median_arm_length",
    "analytic_relative_position_series",
    "analytic_relative_positions",
    "analytic_ssb_rotation_matrix",
    "relabel_raw_ltt_to_standard",
    "relabel_raw_spacecraft_to_standard",
    "standard_detector_frame_positions",
]
