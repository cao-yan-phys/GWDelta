"""Shared numerical utilities for the public weak-field examples."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

from gwdelta import (  # noqa: E402
    FastLISAResponseTDI,
    WeakFieldLinkResponse,
    ensure_cuda_dll_directories,
    equal_arm_aet_noise_psd,
    make_orbits_from_spec,
)


SIDEREAL_YEAR_S = 31_558_149.763545603
DAY_S = 86_400.0
SOLAR_EQUATOR_INCLINATION_DEG = 7.25
SOLAR_EQUATOR_ASCENDING_NODE_DEG = 75.76
SOLAR_MASS_KG = 1.98847e30
SOLAR_RADIUS_M = 6.957e8
SOLAR_G1_M2_FREQUENCY_HZ = 2.93e-4
SOLAR_G1_M2_OVERLAP = 5.38e-3
SOLAR_G1_M2_UNIT_DISK_VELOCITY_M_S = 1.96e5
SOLAR_G1_M2_DEFAULT_DISK_VELOCITY_M_S = 1.0e-4


def configure_cuda_if_needed(backend: str) -> None:
    if "cuda" not in backend.lower():
        return
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
    ensure_cuda_dll_directories()


def as_numpy(value) -> np.ndarray:
    getter = getattr(value, "get", None)
    return np.asarray(getter() if callable(getter) else value)


def normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must be a finite nonzero three-vector")
    return vector / norm


def solar_north_pole_ecliptic() -> np.ndarray:
    inclination = np.deg2rad(SOLAR_EQUATOR_INCLINATION_DEG)
    ascending_node = np.deg2rad(SOLAR_EQUATOR_ASCENDING_NODE_DEG)
    return normalized(
        np.asarray(
            [
                np.sin(inclination) * np.sin(ascending_node),
                -np.sin(inclination) * np.cos(ascending_node),
                np.cos(inclination),
            ]
        ),
        name="solar north pole",
    )


def solar_m2_quadrupole(amplitude_kg_m2: float, phase: float) -> np.ndarray:
    if not np.isfinite(amplitude_kg_m2) or amplitude_kg_m2 <= 0.0:
        raise ValueError("solar_quadrupole_kg_m2 must be finite and positive")
    axis = solar_north_pole_ecliptic()
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    p_basis = normalized(np.cross(axis, reference), name="solar mode p basis")
    q_basis = normalized(np.cross(axis, p_basis), name="solar mode q basis")
    helicity_vector = p_basis + 1j * q_basis
    return (
        0.5
        * float(amplitude_kg_m2)
        * np.exp(1j * float(phase))
        * np.outer(helicity_vector, helicity_vector)
    )


def calibrated_solar_g1_m2_quadrupole(
    disk_velocity_m_s: float, phase: float
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Calibrate the solar l=2, m=2, n=-1 g mode to whole-disk velocity."""

    disk_velocity = float(disk_velocity_m_s)
    if not np.isfinite(disk_velocity) or disk_velocity <= 0.0:
        raise ValueError("disk_velocity_m_s must be finite and positive")

    radial_displacement = disk_velocity / SOLAR_G1_M2_UNIT_DISK_VELOCITY_M_S
    harmonic_coefficient = (
        np.sqrt(15.0 / (4.0 * np.pi))
        * SOLAR_MASS_KG
        * SOLAR_RADIUS_M**2
        * SOLAR_G1_M2_OVERLAP
        * radial_displacement
    )
    # Polnarev et al. use D_ab = 3 I_ab.  A circular complex m=2 mode
    # combines their two real m=+/-2 basis tensors, each of norm sqrt(2).
    quadrupole_norm = (2.0 / 3.0) * harmonic_coefficient
    tensor = solar_m2_quadrupole(quadrupole_norm, phase)
    calibration = {
        "solar_model": "updated MESA GS98",
        "solar_mass_kg": SOLAR_MASS_KG,
        "solar_radius_m": SOLAR_RADIUS_M,
        "radial_order": -1,
        "ell": 2,
        "m": 2,
        "frequency_hz": SOLAR_G1_M2_FREQUENCY_HZ,
        "rotation_treatment": (
            "nonrotating eigenfrequency; rotational splitting and pattern "
            "rotation omitted"
        ),
        "quadrupole_overlap_J2": SOLAR_G1_M2_OVERLAP,
        "unit_displacement_disk_velocity_m_s": (
            SOLAR_G1_M2_UNIT_DISK_VELOCITY_M_S
        ),
        "assumed_disk_velocity_m_s": disk_velocity,
        "dimensionless_radial_displacement": radial_displacement,
        "radial_surface_displacement_m": radial_displacement * SOLAR_RADIUS_M,
        "quadrupole_frobenius_norm_kg_m2": quadrupole_norm,
        "quadrupole_convention": (
            "I_ab=D_ab/3; circular complex m=2 Frobenius normalization"
        ),
        "mode_source": "https://arxiv.org/abs/2602.18385",
        "amplitude_source": "https://arxiv.org/abs/astro-ph/9512091",
        "disk_integration_source": "https://arxiv.org/abs/1210.5525",
    }
    return tensor, calibration


def build_time_grids(
    years: float, dt_s: float, t_buffer_s: float
) -> tuple[np.ndarray, np.ndarray, int]:
    if not np.isfinite(years) or not np.isfinite(dt_s):
        raise ValueError("years and dt must be finite")
    if years <= 0.0 or dt_s <= 0.0:
        raise ValueError("years and dt must be positive")
    if not np.isfinite(t_buffer_s) or t_buffer_s <= 0.0:
        raise ValueError("t_buffer must be finite and positive")
    analysis_samples = int(np.floor(years * SIDEREAL_YEAR_S / dt_s))
    trim_samples = int(t_buffer_s / dt_s)
    if analysis_samples < 16 or trim_samples < 1:
        raise ValueError("time grid or TDI buffer is too short")
    response_samples = analysis_samples + 2 * trim_samples
    guard_samples = max(2, int(np.ceil(100.0 / dt_s)))
    motion_time = np.arange(guard_samples + response_samples, dtype=float) * dt_s
    response_time = motion_time[guard_samples:]
    return motion_time, response_time, analysis_samples


def make_esa_lisa_orbits(duration_s: float, orbit_dt_s: float, response_backend: str):
    """Build the numerical ESA LISA orbit at its native epoch and orientation."""

    return make_orbits_from_spec(
        {
            "base": "esa",
            "orbit_dt": float(orbit_dt_s),
            "use_project_phase_defaults": False,
        },
        duration=float(duration_s),
        force_backend=response_backend,
    )


def make_response_engines(
    orbits,
    *,
    response_backend: str,
    quadrature_order: int,
    chunk_size: int,
    tdi_order: int,
    t_buffer_s: float,
) -> tuple[WeakFieldLinkResponse, FastLISAResponseTDI]:
    link_engine = WeakFieldLinkResponse(
        orbits=orbits,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
        force_backend=response_backend,
    )
    tdi_engine = FastLISAResponseTDI(
        orbits=orbits,
        order=tdi_order,
        tdi="2nd generation",
        tdi_chan="AE",
        force_backend=response_backend,
        t_buffer=t_buffer_s,
        trim_garbage=True,
    )
    return link_engine, tdi_engine


def tdi_channels_numpy(result) -> dict[str, np.ndarray]:
    return {
        "t": np.asarray(result.t, dtype=float),
        "A": as_numpy(result.channels["A"]).astype(float, copy=False),
        "E": as_numpy(result.channels["E"]).astype(float, copy=False),
    }


def channel_spectra(
    time_s: np.ndarray, channels: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or len(time_s) < 2:
        raise ValueError("spectrum time grid must contain at least two samples")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("spectrum time grid must be finite")
    dt_s = float(np.median(np.diff(time_s)))
    if dt_s <= 0.0 or not np.allclose(
        np.diff(time_s), dt_s, rtol=1.0e-11, atol=1.0e-10
    ):
        raise ValueError("spectrum time grid must be strictly increasing and uniform")
    frequency_hz = np.fft.rfftfreq(len(time_s), d=dt_s)
    spectra = {}
    for channel in ("A", "E"):
        values = np.asarray(channels[channel], dtype=float)
        if values.shape != time_s.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{channel} must be a finite array matching the time grid")
        spectra[channel] = dt_s * np.fft.rfft(values)
    return frequency_hz, spectra


def compute_ae_snr(
    time_s: np.ndarray, channels: dict[str, np.ndarray]
) -> dict[str, object]:
    time_s = np.asarray(time_s, dtype=float)
    dt_s = float(np.median(np.diff(time_s)))
    if not np.allclose(np.diff(time_s), dt_s, rtol=1.0e-11, atol=1.0e-10):
        raise ValueError("SNR input time grid must be uniform")
    frequency_hz, spectra = channel_spectra(time_s, channels)
    df_hz = float(frequency_hz[1] - frequency_hz[0])
    psd = equal_arm_aet_noise_psd(
        frequency_hz,
        "lisa",
        channels="AE",
        tdi_generation="second",
    )
    positive = frequency_hz > 0.0
    weights = np.full(np.count_nonzero(positive), 4.0)
    if len(time_s) % 2 == 0:
        weights[-1] = 2.0
    snr2 = {}
    for index, channel in enumerate(("A", "E")):
        integrand = np.abs(spectra[channel][positive]) ** 2 / psd[index, positive]
        snr2[channel] = float(df_hz * np.sum(weights * integrand))
    return {
        "frequency_hz": frequency_hz,
        "spectra": spectra,
        "psd": psd,
        "snr_A": float(np.sqrt(snr2["A"])),
        "snr_E": float(np.sqrt(snr2["E"])),
        "snr_AE": float(np.sqrt(snr2["A"] + snr2["E"])),
        "snr2_A": snr2["A"],
        "snr2_E": snr2["E"],
        "df_hz": df_hz,
        "fft_convention": "dt * rfft, rectangular observation, no window",
        "psd_convention": "one-sided equal-arm LISA second-generation S_A=S_E=S_EA",
        "noise_scope": "instrumental noise only; no Galactic confusion foreground",
    }


def display_scale(*arrays: np.ndarray) -> tuple[float, int]:
    maximum = max(float(np.max(np.abs(array))) for array in arrays)
    if maximum == 0.0 or not np.isfinite(maximum):
        return 1.0, 0
    exponent = int(3 * np.floor(np.log10(maximum) / 3.0))
    return 10.0**exponent, exponent


def configure_latex_roman() -> None:
    """Use LaTeX Computer Modern Roman, with a mathtext fallback."""

    import matplotlib.pyplot as plt

    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": use_tex,
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
        }
    )
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
