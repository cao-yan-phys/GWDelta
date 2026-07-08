"""One-year eccentric AK waveform through second-generation Taiji A/E TDI.

The AK waveform generator is intentionally kept outside the public ``gwdelta``
source tree.  This example imports it from the sibling ``benchmark_waveforms``
workspace directory, then uses GWDelta only for orbit construction and TDI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (SRC_ROOT, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def preconfigure_cuda_environment() -> None:
    if os.name != "nt":
        return
    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3")
    if not cuda_root.exists():
        return
    for key in ("CUDA_PATH", "CUDA_PATH_V12_3", "CUDA_HOME", "CUPY_CUDA_PATH", "CUDAToolkit_ROOT"):
        os.environ.setdefault(key, str(cuda_root))
    preferred = [cuda_root / "bin", cuda_root / "lib" / "x64", cuda_root / "lib"]
    preferred_resolved = [str(path.resolve()) for path in preferred if path.exists()]
    preferred_norm = {path.lower() for path in preferred_resolved}
    existing = []
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if not part:
            continue
        try:
            key = str(Path(part).resolve()).lower()
        except OSError:
            key = part.lower()
        if key not in preferred_norm:
            existing.append(part)
    os.environ["PATH"] = os.pathsep.join(preferred_resolved + existing)


preconfigure_cuda_environment()

from gwdelta import (  # noqa: E402
    FastLISAResponseTDI,
    ensure_cuda_dll_directories,
    make_dynamic_equal_arm_orbits_from_reference,
    make_standard_convention_orbits,
    make_orbits_from_spec,
    make_physical_scale,
    select_array_backend,
)
from notebook_waveforms import (  # noqa: E402
    NotebookParameters,
    _notebook_omega_ddot_phase_factor,
    _omega_dot_newtonian,
    _qk_periastron_precession_rate,
    initial_ak_elements,
    initial_state,
)
from pn_qk_waveform import parameters_from_mean_motion_alignment  # noqa: E402


SIDEREAL_YEAR_S = 31558149.763545603
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "taiji_ak_tdi2_1yr_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=8.0)
    parser.add_argument("--response-backend", choices=["cpu", "cuda12x"], default="cuda12x")
    parser.add_argument("--skip-simple", action="store_true")
    parser.add_argument("--skip-static", action="store_true", dest="skip_simple", help=argparse.SUPPRESS)
    parser.add_argument("--skip-realistic", action="store_true")
    parser.add_argument("--skip-pn", action="store_true")
    parser.add_argument("--taiji-orbit-dir", type=Path, default=WORKSPACE_ROOT / "GWDelta_orbit_data" / "MicroSateOrbitEclipticTCB")
    parser.add_argument("--reference-time-s", type=float, default=0.0)
    parser.add_argument("--orbit-dt", type=float, default=600.0)
    parser.add_argument("--orbit-margin-s", type=float, default=1200.0)
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument("--t-buffer", type=float, default=8192.0)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--ak-n-max", type=int, default=10)
    parser.add_argument("--total-mass-solar", type=float, default=80.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--distance-unit", default="Mpc")
    parser.add_argument("--nu", type=float, default=0.234375)
    parser.add_argument("--qk-mean-motion", type=float, default=6.183424846347413e-6)
    parser.add_argument("--eccentricity", type=float, default=0.1)
    parser.add_argument("--qk-u0", type=float, default=-1.0)
    parser.add_argument("--waveform-phi", type=float, default=0.9)
    parser.add_argument("--waveform-theta", type=float, default=0.7)
    parser.add_argument("--integer-pn-factor", type=float, default=1.0)
    parser.add_argument("--boost-factor", type=float, default=1.0)
    parser.add_argument("--pn-global-sign", type=float, default=-1.0)
    parser.add_argument("--spectrum-window-alpha", type=float, default=0.1)
    parser.add_argument("--zoom-time-days", type=float, default=0.005)
    parser.add_argument("--zoom-frequency-bins", type=int, default=6)
    parser.add_argument("--max-plot-points", type=int, default=20000)
    parser.add_argument("--save-npz", action="store_true")
    return parser.parse_args()


def configure_cuda_if_needed(backend: str) -> None:
    if "cuda" not in backend.lower():
        return
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
    if os.name == "nt":
        cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3")
        if cuda_root.exists():
            os.environ.setdefault("CUDA_PATH", str(cuda_root))
            os.environ.setdefault("CUDA_PATH_V12_3", str(cuda_root))
            os.environ.setdefault("CUDA_HOME", str(cuda_root))
            os.environ.setdefault("CUPY_CUDA_PATH", str(cuda_root))
            os.environ.setdefault("CUDAToolkit_ROOT", str(cuda_root))
    ensure_cuda_dll_directories()


def build_time_grid(years: float, dt: float) -> tuple[np.ndarray, float]:
    if years <= 0.0 or dt <= 0.0:
        raise ValueError("years and dt must be positive")
    n_samples = int(np.floor(years * SIDEREAL_YEAR_S / dt))
    if n_samples < 4:
        raise ValueError("time grid is too short")
    t_seconds = np.arange(n_samples, dtype=float) * float(dt)
    return t_seconds, float(dt)


def interpolate_series(t_source: np.ndarray, values: np.ndarray, t_query: float) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(len(t_source), -1)
    out = np.empty(flat.shape[1], dtype=float)
    for i in range(flat.shape[1]):
        out[i] = np.interp(float(t_query), t_source, flat[:, i])
    return out.reshape(values.shape[1:])


def tukey_window(n: int, alpha: float) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=float)
    if alpha <= 0.0:
        return np.ones(n, dtype=float)
    if alpha >= 1.0:
        return np.hanning(n)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n, dtype=float)
    left = x < alpha / 2.0
    right = x >= 1.0 - alpha / 2.0
    w[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[left] - alpha / 2.0)))
    w[right] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[right] - 1.0 + alpha / 2.0)))
    return w


def decimation_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_points, dtype=int))


def to_numpy(values):
    try:
        import cupy as cp

        if isinstance(values, cp.ndarray):
            return cp.asnumpy(values)
    except Exception:
        pass
    return np.asarray(values)


def make_ak_params(args: argparse.Namespace) -> NotebookParameters:
    return NotebookParameters(
        nu=args.nu,
        total_mass=1.0,
        qk_mean_motion=args.qk_mean_motion,
        qk_e_t=args.eccentricity,
        qk_u0=args.qk_u0,
        phis=args.waveform_phi,
        thetas=args.waveform_theta,
        integer_pn_factor=args.integer_pn_factor,
        boost_factor=args.boost_factor,
    )


def component_masses_solar(total_mass_solar: float, nu: float) -> tuple[float, float]:
    delta = np.sqrt(max(0.0, 1.0 - 4.0 * nu))
    m1 = 0.5 * total_mass_solar * (1.0 + delta)
    m2 = 0.5 * total_mass_solar * (1.0 - delta)
    return float(m1), float(m2)


def ak_f22_frequency_summary(params: NotebookParameters, scale, t_code_end: float) -> dict[str, float]:
    init = initial_state(params)
    ak = initial_ak_elements(params)
    precession_rate = params.integer_pn_factor * _qk_periastron_precession_rate(init, params)
    omega_dot = _omega_dot_newtonian(ak.mean_motion, ak.eccentricity, params.nu, params.boost_factor)
    omega_ddot_phase = _notebook_omega_ddot_phase_factor(init, params)
    omega_phi_start = init.omega0 + precession_rate
    omega_phi_end = omega_phi_start + omega_dot * float(t_code_end) + 0.5 * omega_ddot_phase * float(t_code_end) ** 2
    return {
        "radial_hz": float(init.omega0 / (2.0 * np.pi * scale.time_unit_s)),
        "orbital_hz": float(omega_phi_start / (2.0 * np.pi * scale.time_unit_s)),
        "f22_hz": float(omega_phi_start / (np.pi * scale.time_unit_s)),
        "f22_start_hz": float(omega_phi_start / (np.pi * scale.time_unit_s)),
        "f22_end_hz": float(omega_phi_end / (np.pi * scale.time_unit_s)),
        "omega_dot_code": float(omega_dot),
        "omega_ddot_phase_code": float(omega_ddot_phase),
    }


def generate_ak_polarizations(t_seconds: np.ndarray, args: argparse.Namespace):
    params = make_ak_params(args)
    scale = make_physical_scale(
        total_mass_solar=args.total_mass_solar,
        distance=args.distance,
        distance_unit=args.distance_unit,
        code_total_mass=params.total_mass,
    )
    t_code = t_seconds / scale.time_unit_s
    if args.response_backend == "cuda12x":
        from benchmark_waveforms.cupy_ak_waveforms import sample_ak_polarizations_fourier_raw_cuda

        samples = sample_ak_polarizations_fourier_raw_cuda(t_code, params, n_max=args.ak_n_max)
        h_plus = samples.h_plus * scale.strain_scale
        h_cross = samples.h_cross * scale.strain_scale
        sample_backend = "cupy_raw"
    else:
        from benchmark_waveforms.waveforms import sample_ak_polarizations_fourier

        backend = select_array_backend(force="cpu")
        samples = sample_ak_polarizations_fourier(t_code, params, n_max=args.ak_n_max, backend=backend)
        h_plus = np.asarray(samples.h_plus, dtype=float) * scale.strain_scale
        h_cross = np.asarray(samples.h_cross, dtype=float) * scale.strain_scale
        sample_backend = "numpy"

    frequencies = ak_f22_frequency_summary(params, scale, float(t_code[-1]))
    m1_solar, m2_solar = component_masses_solar(args.total_mass_solar, args.nu)
    meta = {
        "model": "external_benchmark_AK_fourier",
        "sample_backend": sample_backend,
        "ak_n_max": int(args.ak_n_max),
        "params": asdict(params),
        "scale": {
            "time_unit_s": float(scale.time_unit_s),
            "strain_scale": float(scale.strain_scale),
            "total_mass_solar": float(args.total_mass_solar),
            "component_masses_solar": [m1_solar, m2_solar],
            "distance": float(args.distance),
            "distance_unit": args.distance_unit,
        },
        "frequencies": frequencies,
    }
    return h_plus, h_cross, meta


def generate_pn_polarizations(t_seconds: np.ndarray, args: argparse.Namespace):
    ak_params = make_ak_params(args)
    scale = make_physical_scale(
        total_mass_solar=args.total_mass_solar,
        distance=args.distance,
        distance_unit=args.distance_unit,
        code_total_mass=ak_params.total_mass,
    )
    init = initial_state(ak_params)
    pn_params = parameters_from_mean_motion_alignment(
        mean_motion=args.qk_mean_motion,
        e_t=args.eccentricity,
        nu=args.nu,
        total_mass=ak_params.total_mass,
        initial_eccentric_anomaly=args.qk_u0,
        initial_orbital_phase=init.theta0,
        distance=1.0,
        t0=0.0,
        global_sign=args.pn_global_sign,
    )
    backend = select_array_backend(force="cupy" if args.response_backend == "cuda12x" else "cpu")
    from benchmark_waveforms.waveforms import sample_evolving_pn_qk

    t_code = t_seconds / scale.time_unit_s
    samples = sample_evolving_pn_qk(
        t_code,
        pn_params,
        theta=args.waveform_theta,
        phi=args.waveform_phi,
        backend=backend,
        include_h20=True,
        rtol=1e-9,
        atol=(1e-13, 1e-13, 1e-9, 1e-9),
    )
    if not samples.metadata.get("phase_success", True):
        raise RuntimeError(str(samples.metadata.get("phase_message", "PN phase evolution failed")))
    h_plus = samples.h_plus * scale.strain_scale
    h_cross = samples.h_cross * scale.strain_scale
    omega = float(pn_params.x0**1.5 / pn_params.total_mass)
    k = float(3.0 * pn_params.x0 / (1.0 - pn_params.e_t0**2))
    m1_solar, m2_solar = component_masses_solar(args.total_mass_solar, args.nu)
    meta = {
        "model": "external_benchmark_PN_1PN_evolving_0PN_modes_aligned",
        "sample_backend": backend.name,
        "generator_metadata": {
            **samples.metadata,
            "model": str(samples.metadata.get("model", "PN_evolving_phases_backend_modes")).replace("PN" + "_QK", "PN"),
        },
        "params": asdict(pn_params),
        "alignment": {
            "mean_motion": float(args.qk_mean_motion),
            "e_t": float(args.eccentricity),
            "initial_eccentric_anomaly": float(args.qk_u0),
            "initial_orbital_phase": float(init.theta0),
            "global_sign": float(args.pn_global_sign),
        },
        "scale": {
            "time_unit_s": float(scale.time_unit_s),
            "strain_scale": float(scale.strain_scale),
            "total_mass_solar": float(args.total_mass_solar),
            "component_masses_solar": [m1_solar, m2_solar],
            "distance": float(args.distance),
            "distance_unit": args.distance_unit,
        },
        "frequencies": {
            "mean_motion_hz": float((omega / (1.0 + k)) / (2.0 * np.pi * scale.time_unit_s)),
            "orbital_hz": float(omega / (2.0 * np.pi * scale.time_unit_s)),
            "f22_hz": float(omega / (np.pi * scale.time_unit_s)),
        },
    }
    return h_plus, h_cross, meta


def selected_orbit_labels(args: argparse.Namespace) -> list[str]:
    labels: list[str] = []
    if not args.skip_simple:
        labels.append("simple")
    if not args.skip_realistic:
        labels.append("realistic")
    if not labels:
        raise ValueError("at least one orbit must be enabled")
    return labels


def make_realistic_taiji_orbits(args: argparse.Namespace, duration_s: float):
    if not args.taiji_orbit_dir.exists():
        raise FileNotFoundError(f"Taiji orbit directory does not exist: {args.taiji_orbit_dir}")
    raw_orbits = make_orbits_from_spec(
        {
            "base": "taiji-accurate",
            "orbit_dir": str(args.taiji_orbit_dir),
            "orbit_dt": float(args.orbit_dt),
        },
        duration=duration_s,
        force_backend=args.response_backend,
    )
    return make_standard_convention_orbits(raw_orbits, force_backend=args.response_backend)


def match_to_dict(match) -> dict[str, object]:
    raw = asdict(match)
    return {
        key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
        for key, value in raw.items()
    }


def build_orbit_map(args: argparse.Namespace, duration_s: float) -> tuple[dict[str, object], dict[str, object]]:
    realistic_orbits = make_realistic_taiji_orbits(args, duration_s)
    reference_time = float(args.reference_time_s)
    t_base = np.asarray(realistic_orbits.t_base, dtype=float)
    if reference_time < t_base[0] or reference_time > t_base[-1]:
        raise ValueError("reference_time_s must lie inside the sampled realistic Taiji orbit")
    reference_positions = interpolate_series(t_base, realistic_orbits.x_base, reference_time)
    simple_orbits, simple_match = make_dynamic_equal_arm_orbits_from_reference(
        reference_positions,
        duration_s=duration_s,
        reference_time_s=reference_time,
        orbit_dt=float(args.orbit_dt),
        force_backend=args.response_backend,
    )
    orbit_map: dict[str, object] = {}
    if not args.skip_simple:
        orbit_map["simple"] = simple_orbits
    if not args.skip_realistic:
        orbit_map["realistic"] = realistic_orbits
    alignment = {
        "rule": "dynamic simple equal-arm orbit is matched to the standard-labeled realistic Taiji triangle at reference_time_s",
        "reference_time_s": reference_time,
        "simple_equal_arm_match": match_to_dict(simple_match),
    }
    return orbit_map, alignment


def compute_tdi_for_orbit(
    *,
    label: str,
    orbits,
    args: argparse.Namespace,
    t_seconds: np.ndarray,
    h_plus,
    h_cross,
) -> tuple[dict[str, object], float]:
    tic = time.perf_counter()
    response = FastLISAResponseTDI(
        orbits=orbits,
        order=args.order,
        tdi="2nd generation",
        tdi_chan="AE",
        force_backend=args.response_backend,
        t_buffer=args.t_buffer,
        trim_garbage=True,
        cache_response=True,
    )
    tdi_result = response.compute(t_seconds, h_plus, h_cross, lam=args.lam, beta=args.beta, t0=0.0)
    elapsed = time.perf_counter() - tic
    A_t = tdi_result.channels["A"]
    E_t = tdi_result.channels["E"]
    t_tdi = np.asarray(tdi_result.t, dtype=float)
    return {
        "t": t_tdi,
        "A": A_t,
        "E": E_t,
        "metadata": dict(tdi_result.metadata),
    }, elapsed


def compute_spectrum(values, dt: float, alpha: float):
    try:
        import cupy as cp

        if isinstance(values, cp.ndarray):
            window = cp.asarray(tukey_window(int(values.size), alpha), dtype=cp.float64)
            freqs = cp.asarray(np.fft.rfftfreq(int(values.size), d=dt))
            spectrum = dt * cp.fft.rfft(values * window)
            return cp.asnumpy(freqs), cp.asnumpy(cp.abs(spectrum))
    except Exception:
        pass
    arr = np.asarray(values, dtype=float)
    freqs = np.fft.rfftfreq(len(arr), d=dt)
    spectrum = dt * np.fft.rfft(arr * tukey_window(len(arr), alpha))
    return freqs, np.abs(spectrum)


def response_plot_styles() -> dict[str, dict[str, object]]:
    return {
        "simple": {"label": "simple Taiji orbit (AK)", "ls": "-", "lw": 0.9, "color": None},
        "realistic": {"label": "realistic Taiji orbit (AK)", "ls": "--", "lw": 0.9, "color": None},
        "pn_realistic": {"label": "realistic Taiji orbit (PN)", "ls": "--", "lw": 1.0, "color": "0.0"},
    }


def limited_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    if indices.size <= max_points:
        return indices
    pick = np.linspace(0, indices.size - 1, int(max_points), dtype=int)
    return indices[pick]


def plot_outputs(
    *,
    t_seconds: np.ndarray,
    h_plus,
    h_cross,
    responses: dict[str, dict[str, object]],
    f22_start_hz: float | None,
    f22_end_hz: float | None,
    figure_path: Path,
    max_plot_points: int,
) -> None:
    import matplotlib.pyplot as plt

    h_idx = decimation_indices(len(t_seconds), max_plot_points)
    h_plus_plot = to_numpy(h_plus[h_idx])
    h_cross_plot = to_numpy(h_cross[h_idx])
    styles = response_plot_styles()

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig = plt.figure(figsize=(11.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_wave = fig.add_subplot(gs[0, :])
    ax_a_time = fig.add_subplot(gs[1, 0])
    ax_e_time = fig.add_subplot(gs[1, 1])
    ax_a_freq = fig.add_subplot(gs[2, 0])
    ax_e_freq = fig.add_subplot(gs[2, 1])

    ax_wave.plot(t_seconds[h_idx] / SIDEREAL_YEAR_S, h_plus_plot, lw=0.8, label=r"$h_+$")
    ax_wave.plot(t_seconds[h_idx] / SIDEREAL_YEAR_S, h_cross_plot, lw=0.8, label=r"$h_\times$")
    ax_wave.set_xlabel(r"$t$ [yr]")
    ax_wave.set_ylabel(r"$h(t)$")
    ax_wave.grid(alpha=0.25)
    ax_wave.legend(loc="best")

    for label, data in responses.items():
        style = styles.get(label, {"label": label, "ls": "-", "lw": 0.9, "color": None})
        t_tdi = np.asarray(data["t"], dtype=float)
        tdi_idx = decimation_indices(len(t_tdi), max_plot_points)
        ax_a_time.plot(
            t_tdi[tdi_idx] / SIDEREAL_YEAR_S,
            to_numpy(data["A"][tdi_idx]),
            linestyle=style["ls"],
            lw=style["lw"],
            color=style["color"],
            label=style["label"],
        )
        ax_e_time.plot(
            t_tdi[tdi_idx] / SIDEREAL_YEAR_S,
            to_numpy(data["E"][tdi_idx]),
            linestyle=style["ls"],
            lw=style["lw"],
            color=style["color"],
            label=style["label"],
        )

        freqs = np.asarray(data["freqs"], dtype=float)
        positive = freqs > 0.0
        ax_a_freq.loglog(
            freqs[positive],
            np.asarray(data["A_abs"])[positive],
            linestyle=style["ls"],
            lw=style["lw"],
            color=style["color"],
            label=style["label"],
        )
        ax_e_freq.loglog(
            freqs[positive],
            np.asarray(data["E_abs"])[positive],
            linestyle=style["ls"],
            lw=style["lw"],
            color=style["color"],
            label=style["label"],
        )

    for ax, ylabel in ((ax_a_time, r"$A(t)$"), (ax_e_time, r"$E(t)$")):
        ax.set_xlabel(r"$t$ [yr]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    for ax, ylabel in ((ax_a_freq, r"$|\tilde A(f)|$"), (ax_e_freq, r"$|\tilde E(f)|$")):
        if f22_start_hz is not None and f22_start_hz > 0.0:
            ax.axvline(f22_start_hz, color="0.25", ls="-.", lw=0.9, label="f22_start")
        if f22_end_hz is not None and f22_end_hz > 0.0:
            ax.axvline(f22_end_hz, color="0.45", ls=":", lw=1.0, label="f22_end")
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="best")

    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_a_zoom_outputs(
    *,
    responses: dict[str, dict[str, object]],
    f22_start_hz: float | None,
    f22_end_hz: float | None,
    figure_path: Path,
    zoom_time_days: float,
    zoom_frequency_bins: int,
    max_plot_points: int,
) -> None:
    import matplotlib.pyplot as plt

    styles = response_plot_styles()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    ax_start, ax_end, ax_freq = axes

    t_min = min(float(np.asarray(data["t"], dtype=float)[0]) for data in responses.values())
    t_max = max(float(np.asarray(data["t"], dtype=float)[-1]) for data in responses.values())
    zoom_s = max(float(zoom_time_days), 0.0) * 86400.0
    if zoom_s <= 0.0:
        zoom_s = min(0.25 * 86400.0, t_max - t_min)
    windows = ((t_min, min(t_min + zoom_s, t_max), ax_start), (max(t_min, t_max - zoom_s), t_max, ax_end))

    for lo, hi, ax in windows:
        for label, data in responses.items():
            style = styles.get(label, {"label": label, "ls": "-", "lw": 0.9, "color": None})
            t_tdi = np.asarray(data["t"], dtype=float)
            idx = np.flatnonzero((t_tdi >= lo) & (t_tdi <= hi))
            idx = limited_indices(idx, max_plot_points)
            if idx.size == 0:
                continue
            ax.plot(
                t_tdi[idx] / 86400.0,
                to_numpy(data["A"][idx]),
                linestyle=style["ls"],
                lw=style["lw"],
                color=style["color"],
                label=style["label"],
            )
        ax.set_xlabel(r"$t$ [d]")
        ax.set_ylabel(r"$A(t)$")
        ax.set_xlim(lo / 86400.0, hi / 86400.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    f22_values = [
        float(value)
        for value in (f22_start_hz, f22_end_hz)
        if value is not None and float(value) > 0.0
    ]
    first_freqs = np.asarray(next(iter(responses.values()))["freqs"], dtype=float)
    positive_freqs = first_freqs[first_freqs > 0.0]
    df = float(np.median(np.diff(positive_freqs))) if positive_freqs.size > 1 else 0.0
    if f22_values and df > 0.0:
        pad = max(1, int(zoom_frequency_bins)) * df
        fmin = max(float(positive_freqs[0]), min(f22_values) - pad)
        fmax = max(f22_values) + pad
    else:
        fmin, fmax = 1.0e-4, 1.0e-3
    freq_scale = 1.0e3
    for label, data in responses.items():
        style = styles.get(label, {"label": label, "ls": "-", "lw": 0.9, "color": None})
        freqs = np.asarray(data["freqs"], dtype=float)
        values = np.asarray(data["A_abs"], dtype=float)
        idx = np.flatnonzero((freqs >= fmin) & (freqs <= fmax) & (values > 0.0))
        idx = limited_indices(idx, max_plot_points)
        if idx.size == 0:
            continue
        ax_freq.semilogy(
            freq_scale * freqs[idx],
            values[idx],
            linestyle=style["ls"],
            lw=style["lw"],
            color=style["color"],
            label=style["label"],
        )
    if f22_start_hz is not None and f22_start_hz > 0.0:
        ax_freq.axvline(freq_scale * f22_start_hz, color="0.25", ls="-.", lw=0.9, label="f22_start")
    if f22_end_hz is not None and f22_end_hz > 0.0:
        ax_freq.axvline(freq_scale * f22_end_hz, color="0.45", ls=":", lw=1.0, label="f22_end")
    ax_freq.set_xlim(freq_scale * fmin, freq_scale * fmax)
    ax_freq.set_xlabel(r"$f$ [mHz]")
    ax_freq.set_ylabel(r"$|\tilde A(f)|$")
    ax_freq.grid(alpha=0.25, which="both")
    ax_freq.legend(loc="best")

    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    def log(message: str) -> None:
        print(f"[taiji_ak_tdi2_1yr_demo] {message}", flush=True)

    tic_total = time.perf_counter()
    log("configuring CUDA/runtime")
    configure_cuda_if_needed(args.response_backend)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "taiji_ak_tdi2_1yr_demo.png"
    a_zoom_figure_path = output_dir / "taiji_ak_tdi2_1yr_demo_A_zoom.png"
    summary_path = output_dir / "summary.json"
    npz_path = output_dir / "taiji_ak_tdi2_1yr_demo_decimated.npz"

    t_seconds, dt = build_time_grid(args.years, args.dt)
    log(f"built time grid: samples={len(t_seconds)}, dt={dt:g} s, years={args.years:g}")
    tic = time.perf_counter()
    log("generating external CUDA AK waveform")
    h_plus, h_cross, waveform_meta = generate_ak_polarizations(t_seconds, args)
    waveform_s = time.perf_counter() - tic
    log(f"waveform done in {waveform_s:.3g} s")
    pn_h_plus = None
    pn_h_cross = None
    pn_waveform_meta = None
    pn_waveform_s = None
    if not args.skip_pn:
        tic = time.perf_counter()
        log("generating aligned PN waveform")
        pn_h_plus, pn_h_cross, pn_waveform_meta = generate_pn_polarizations(t_seconds, args)
        pn_waveform_s = time.perf_counter() - tic
        log(f"PN waveform done in {pn_waveform_s:.3g} s")

    duration_s = float(t_seconds[-1] - t_seconds[0] + args.orbit_margin_s)
    responses: dict[str, dict[str, object]] = {}
    response_timings: dict[str, float] = {}
    log("building realistic Taiji orbit and dynamic simple Taiji orbit")
    tic = time.perf_counter()
    orbit_map, orbit_alignment = build_orbit_map(args, duration_s)
    orbit_setup_s = time.perf_counter() - tic
    log(f"orbit setup done in {orbit_setup_s:.3g} s")
    orbit_labels = list(orbit_map)
    for label, orbits in orbit_map.items():
        log(f"computing {label} Taiji A/E response")
        response_data, response_s = compute_tdi_for_orbit(
            label=label,
            orbits=orbits,
            args=args,
            t_seconds=t_seconds,
            h_plus=h_plus,
            h_cross=h_cross,
        )
        responses[label] = response_data
        response_timings[label] = response_s
        log(f"{label} response done in {response_s:.3g} s")
    if pn_h_plus is not None and pn_h_cross is not None and "realistic" in orbit_map:
        label = "pn_realistic"
        log("computing PN realistic Taiji A/E response")
        response_data, response_s = compute_tdi_for_orbit(
            label=label,
            orbits=orbit_map["realistic"],
            args=args,
            t_seconds=t_seconds,
            h_plus=pn_h_plus,
            h_cross=pn_h_cross,
        )
        responses[label] = response_data
        response_timings[label] = response_s
        log(f"PN realistic response done in {response_s:.3g} s")

    tic = time.perf_counter()
    log("computing spectra")
    for label, data in responses.items():
        freqs, A_abs = compute_spectrum(data["A"], dt, args.spectrum_window_alpha)
        _, E_abs = compute_spectrum(data["E"], dt, args.spectrum_window_alpha)
        data["freqs"] = freqs
        data["A_abs"] = A_abs
        data["E_abs"] = E_abs
    spectrum_s = time.perf_counter() - tic

    log("plotting outputs")
    plot_outputs(
        t_seconds=t_seconds,
        h_plus=h_plus,
        h_cross=h_cross,
        responses=responses,
        f22_start_hz=waveform_meta["frequencies"].get("f22_start_hz"),
        f22_end_hz=waveform_meta["frequencies"].get("f22_end_hz"),
        figure_path=figure_path,
        max_plot_points=args.max_plot_points,
    )
    plot_a_zoom_outputs(
        responses=responses,
        f22_start_hz=waveform_meta["frequencies"].get("f22_start_hz"),
        f22_end_hz=waveform_meta["frequencies"].get("f22_end_hz"),
        figure_path=a_zoom_figure_path,
        zoom_time_days=args.zoom_time_days,
        zoom_frequency_bins=args.zoom_frequency_bins,
        max_plot_points=args.max_plot_points,
    )

    summary = {
        "waveform": waveform_meta,
        "pn_waveform": pn_waveform_meta,
        "time_grid": {
            "years": float(args.years),
            "dt_s": float(dt),
            "samples": int(len(t_seconds)),
            "duration_s": float(t_seconds[-1] - t_seconds[0]),
        },
        "response": {
            "backend": args.response_backend,
            "tdi": "2nd generation",
            "tdi_chan": "AE",
            "orbit_labels": orbit_labels,
            "response_labels": list(responses),
            "taiji_orbit_dir": str(args.taiji_orbit_dir),
            "orbit_alignment": orbit_alignment,
            "metadata": {label: data["metadata"] for label, data in responses.items()},
        },
        "spectrum": {
            "window": "tukey",
            "window_alpha": float(args.spectrum_window_alpha),
            "frequency_bins": {label: int(len(data["freqs"])) for label, data in responses.items()},
            "df_hz": {
                label: float(data["freqs"][1] - data["freqs"][0]) if len(data["freqs"]) > 1 else None
                for label, data in responses.items()
            },
        },
        "zoom": {
            "time_days": float(args.zoom_time_days),
            "frequency_bins_each_side": int(args.zoom_frequency_bins),
            "a_zoom_figure": str(a_zoom_figure_path),
        },
        "timings_s": {
            "waveform": float(waveform_s),
            "pn_waveform": None if pn_waveform_s is None else float(pn_waveform_s),
            "orbit_setup": float(orbit_setup_s),
            "response": {label: float(value) for label, value in response_timings.items()},
            "spectrum": float(spectrum_s),
            "total": float(time.perf_counter() - tic_total),
        },
        "outputs": {
            "figure": str(figure_path),
            "a_zoom_figure": str(a_zoom_figure_path),
            "summary": str(summary_path),
            "npz": str(npz_path) if args.save_npz else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    if args.save_npz:
        h_idx = decimation_indices(len(t_seconds), args.max_plot_points)
        arrays: dict[str, object] = {
            "t_seconds": t_seconds[h_idx],
            "h_plus": to_numpy(h_plus[h_idx]),
            "h_cross": to_numpy(h_cross[h_idx]),
            "summary_json": np.asarray(json.dumps(summary, separators=(",", ":"), default=str)),
        }
        if pn_h_plus is not None and pn_h_cross is not None:
            arrays["pn_h_plus"] = to_numpy(pn_h_plus[h_idx])
            arrays["pn_h_cross"] = to_numpy(pn_h_cross[h_idx])
        for label, data in responses.items():
            t_tdi = np.asarray(data["t"], dtype=float)
            tdi_idx = decimation_indices(len(t_tdi), args.max_plot_points)
            arrays[f"{label}_t_tdi"] = t_tdi[tdi_idx]
            arrays[f"{label}_A"] = to_numpy(data["A"][tdi_idx])
            arrays[f"{label}_E"] = to_numpy(data["E"][tdi_idx])
            arrays[f"{label}_freqs"] = data["freqs"]
            arrays[f"{label}_A_abs"] = data["A_abs"]
            arrays[f"{label}_E_abs"] = data["E_abs"]
        np.savez(npz_path, **arrays)
    log("done")
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
