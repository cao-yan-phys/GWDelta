"""Compute the SNR of a monochromatic elliptically polarized source.

This script keeps the long-duration detector response in the time domain.  It
then Fourier-transforms the TDI channels and weights them with the equal-arm
instrumental-noise PSDs from ``gwdelta.noise``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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
    DETECTOR_ORDER,
    DetectorNetwork,
    ensure_cuda_dll_directories,
    equal_arm_aet_noise_psd,
    get_noise_model,
    load_orbit_overrides,
    parse_detector_names,
    select_array_backend,
)


SIDEREAL_YEAR_S = 31_558_149.763545603
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "monochromatic_snr_time_domain" / "summary.json"
TDI_CHANNELS = "AE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=30.0, help="time-domain sample spacing in seconds")
    parser.add_argument("--frequency", "--f0", type=float, default=3.0e-3, help="source frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=1.0e-22, help="plus-polarization strain amplitude")
    parser.add_argument("--ellipticity", type=float, default=1.0, help="h_cross/h_plus amplitude ratio")
    parser.add_argument("--phase", type=float, default=0.0, help="initial plus-polarization phase in radians")
    parser.add_argument(
        "--cross-phase",
        type=float,
        default=-0.5 * np.pi,
        help="extra h_cross phase relative to h_plus in radians",
    )
    parser.add_argument("--lam", type=float, default=0.3, help="ecliptic longitude in radians")
    parser.add_argument("--beta", type=float, default=0.4, help="ecliptic latitude in radians")
    parser.add_argument(
        "--detectors",
        default="taiji",
        help="comma-separated detector list, or 'all' for lisa,taiji,tianqin,bbo",
    )
    parser.add_argument("--tdi-generation", choices=["first", "second"], default="second")
    parser.add_argument("--response-backend", choices=["cpu", "cuda12x"], default="cuda12x")
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument(
        "--t-buffer",
        type=float,
        default=None,
        help="response buffer in seconds; omitted means auto-estimate from dt, order, and orbit size",
    )
    parser.add_argument("--trim-garbage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orbit-dt", type=float, default=600.0)
    parser.add_argument("--orbit-margin-s", type=float, default=1200.0)
    parser.add_argument(
        "--orbit-config-json",
        type=Path,
        default=None,
        help="optional JSON object with per-detector OrbitSpec overrides for the initial configuration",
    )
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--no-output-json", action="store_true")
    return parser.parse_args()


def configure_cuda_if_needed(backend: str) -> None:
    if "cuda" not in backend.lower():
        return
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
    ensure_cuda_dll_directories()


def build_time_grid(years: float, dt: float) -> np.ndarray:
    if years <= 0.0 or dt <= 0.0:
        raise ValueError("years and dt must be positive")
    n_samples = int(np.floor(float(years) * SIDEREAL_YEAR_S / float(dt)))
    if n_samples < 4:
        raise ValueError("time grid has fewer than four samples")
    return np.arange(n_samples, dtype=float) * float(dt)


def generate_polarizations(t: np.ndarray, args: argparse.Namespace):
    nyquist = 0.5 / float(args.dt)
    if not (0.0 < args.frequency < nyquist):
        raise ValueError(f"frequency must lie between 0 and Nyquist={nyquist:g} Hz")

    backend = select_array_backend(force="cupy" if args.response_backend == "cuda12x" else "cpu")
    t_backend = backend.asarray(t, dtype=float)
    phase = 2.0 * np.pi * float(args.frequency) * t_backend + float(args.phase)
    h_plus = float(args.amplitude) * backend.xp.cos(phase)
    h_cross = float(args.amplitude) * float(args.ellipticity) * backend.xp.cos(
        phase + float(args.cross_phase)
    )
    return h_plus, h_cross, backend.name


def compute_snr(
    *,
    detector: str,
    channels: dict[str, np.ndarray],
    channel_order: str,
    tdi_generation: str,
) -> dict[str, object]:
    t = np.asarray(channels["t"], dtype=float)
    if t.ndim != 1 or len(t) < 4:
        raise ValueError("TDI time series is too short after trimming")
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1.0e-10, atol=max(1.0e-12, abs(dt) * 1.0e-12)):
        raise ValueError("TDI time grid is not uniformly sampled")

    n = int(len(t))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) < 2:
        raise ValueError("frequency grid is too short")
    df = float(freqs[1] - freqs[0])
    positive = freqs > 0.0
    psd = equal_arm_aet_noise_psd(
        freqs,
        detector,
        channels=channel_order,
        tdi_generation=tdi_generation,
    )

    channel_snr2: dict[str, float] = {}
    peak_frequency: dict[str, float] = {}
    total_snr2 = 0.0
    for idx, channel in enumerate(channel_order):
        values = np.asarray(channels[channel], dtype=float)
        spectrum = dt * np.fft.rfft(values)
        contribution = np.zeros_like(freqs, dtype=float)
        contribution[positive] = (
            4.0
            * df
            * np.abs(spectrum[positive]) ** 2
            / np.asarray(psd[idx][positive], dtype=float)
        )
        snr2 = float(np.sum(contribution[positive]))
        channel_snr2[channel] = snr2
        total_snr2 += snr2
        if np.any(positive):
            peak_idx = int(np.argmax(contribution[positive]))
            peak_frequency[channel] = float(freqs[positive][peak_idx])

    return {
        "snr": float(np.sqrt(max(total_snr2, 0.0))),
        "snr2": float(total_snr2),
        "channel_snr": {key: float(np.sqrt(max(value, 0.0))) for key, value in channel_snr2.items()},
        "channel_snr2": channel_snr2,
        "peak_frequency_hz": peak_frequency,
        "dt_s": dt,
        "samples": n,
        "duration_s": float(t[-1] - t[0] + dt),
        "df_hz": df,
        "noise_model": get_noise_model(detector).name,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    tic_total = time.perf_counter()
    configure_cuda_if_needed(args.response_backend)
    detectors = parse_detector_names(args.detectors)
    orbit_overrides = load_orbit_overrides(args.orbit_config_json)
    t = build_time_grid(args.years, args.dt)
    h_plus, h_cross, waveform_backend = generate_polarizations(t, args)

    network = DetectorNetwork(
        detectors,
        orbit_dt=args.orbit_dt,
        orbit_overrides=orbit_overrides,
        force_backend=args.response_backend,
    )
    network_response = network.compute_response(
        t,
        h_plus,
        h_cross,
        lam=args.lam,
        beta=args.beta,
        tdi_generation=args.tdi_generation,
        tdi_chan=TDI_CHANNELS,
        order=args.order,
        t_buffer=args.t_buffer,
        trim_garbage=args.trim_garbage,
        orbit_margin_s=args.orbit_margin_s,
        skip_unavailable=args.skip_unavailable,
    )
    results: dict[str, object] = {}
    for detector, response in network_response.detectors.items():
        channels = response.as_dict()
        snr = compute_snr(
            detector=detector,
            channels=channels,
            channel_order=TDI_CHANNELS,
            tdi_generation=args.tdi_generation,
        )
        orbit_spec = response.orbit_spec
        results[detector] = {
            "orbit": {
                "base": orbit_spec.base,
                "orbit_dir": None if orbit_spec.orbit_dir is None else str(orbit_spec.orbit_dir),
                "orbit_dt_s": float(orbit_spec.orbit_dt),
                "duration_s": float(orbit_spec.duration),
            },
            "response_metadata": response.metadata,
            "snr": snr,
            "timings_s": {
                "orbit_setup": float(response.orbit_setup_s),
                "response": float(response.response_s),
                "detector_total": float(response.total_s),
            },
        }

    summary = {
        "source": {
            "model": "monochromatic_elliptically_polarized",
            "years": float(args.years),
            "dt_s": float(args.dt),
            "samples": int(len(t)),
            "frequency_hz": float(args.frequency),
            "amplitude": float(args.amplitude),
            "ellipticity": float(args.ellipticity),
            "phase_rad": float(args.phase),
            "cross_phase_rad": float(args.cross_phase),
            "lam_rad": float(args.lam),
            "beta_rad": float(args.beta),
            "waveform_backend": waveform_backend,
        },
        "response": {
            "backend": args.response_backend,
            "tdi_generation": args.tdi_generation,
            "tdi_chan": TDI_CHANNELS,
            "order": int(args.order),
            "requested_t_buffer_s": None if args.t_buffer is None else float(args.t_buffer),
            "trim_garbage": bool(args.trim_garbage),
            "orbit_config_json": None if args.orbit_config_json is None else str(args.orbit_config_json),
            "orbit_overrides": orbit_overrides,
        },
        "detectors_requested": detectors,
        "network_response": network_response.metadata,
        "results": results,
        "failures": network_response.failures,
        "timings_s": {"total": float(time.perf_counter() - tic_total)},
    }
    return summary


def print_summary(summary: dict[str, object]) -> None:
    source = summary["source"]
    print(
        "source: "
        f"f0={source['frequency_hz']:.6g} Hz, "
        f"h0={source['amplitude']:.3e}, "
        f"years={source['years']:.6g}, "
        f"dt={source['dt_s']:.6g} s"
    )
    print("detector  channels  SNR        df [Hz]       duration [d]  backend")
    results = summary["results"]
    for detector in DETECTOR_ORDER:
        if detector not in results:
            continue
        entry = results[detector]
        snr = entry["snr"]
        duration_days = snr["duration_s"] / 86400.0
        backend = entry["response_metadata"].get("backend")
        print(
            f"{detector:<8}  "
            f"{summary['response']['tdi_chan']:<8}  "
            f"{snr['snr']:<10.5g} "
            f"{snr['df_hz']:<12.5g} "
            f"{duration_days:<12.6g} "
            f"{backend}"
        )
    failures = summary.get("failures", {})
    if failures:
        print("failures:")
        for detector, message in failures.items():
            print(f"  {detector}: {message}")


def main() -> None:
    args = parse_args()
    summary = run(args)
    print_summary(summary)
    if not args.no_output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print(f"summary: {args.output_json}")


if __name__ == "__main__":
    main()
