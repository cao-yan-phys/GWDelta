"""Compare realistic Taiji-orbit TD TDI with an analytic static response.

The default waveform file is the WSL-generated SEOBNRv5PHM precessing SMBHB
with l=2 displacement-memory modes.  The response comparison is:

1. realistic Taiji orbit, relabeled from raw orbit-file order to the standard
   LISA/TDI convention, second-generation A/E in the time domain;
2. analytic static equal-arm orbit matched to the realistic orbit at the
   waveform peak, second-generation A/E in the time domain;
3. this project's analytic static frequency-domain second-generation response
   for the same static orbit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (SRC_ROOT,):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from gwdelta import (  # noqa: E402
    FastLISAResponseTDI,
    StaticTaijiFDResponse,
    make_orbits_from_spec,
    make_standard_convention_orbits,
    make_static_equal_arm_orbits_from_reference,
)


DEFAULT_WAVEFORM = (
    REPO_ROOT
    / "examples"
    / "data"
    / "seobnr_v5phm_displacement_memory_waveform.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "taiji_static_tdi2_memory_demo"


def parse_metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in data.files:
        return {}
    raw = data["metadata_json"]
    try:
        return json.loads(str(raw.item()))
    except Exception:
        return json.loads(str(raw))


def load_waveform(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    if "t_seconds" not in data.files:
        raise ValueError("waveform npz must contain t_seconds")
    hp_key = "h_plus_total" if "h_plus_total" in data.files else "h_plus"
    hc_key = "h_cross_total" if "h_cross_total" in data.files else "h_cross"
    if hp_key not in data.files or hc_key not in data.files:
        raise ValueError("waveform npz must contain h_plus/h_cross or h_plus_total/h_cross_total")
    if "h_plus_osc" not in data.files or "h_cross_osc" not in data.files:
        raise ValueError("waveform npz must contain h_plus_osc and h_cross_osc for the no-memory reference")

    t_raw = np.asarray(data["t_seconds"], dtype=float)
    h_plus = np.asarray(data[hp_key], dtype=float)
    h_cross = np.asarray(data[hc_key], dtype=float)
    h_plus_no_memory = np.asarray(data["h_plus_osc"], dtype=float)
    h_cross_no_memory = np.asarray(data["h_cross_osc"], dtype=float)
    n = min(len(t_raw), len(h_plus), len(h_cross), len(h_plus_no_memory), len(h_cross_no_memory))
    t_raw = t_raw[:n]
    h_plus = h_plus[:n]
    h_cross = h_cross[:n]
    h_plus_no_memory = h_plus_no_memory[:n]
    h_cross_no_memory = h_cross_no_memory[:n]
    finite = (
        np.isfinite(t_raw)
        & np.isfinite(h_plus)
        & np.isfinite(h_cross)
        & np.isfinite(h_plus_no_memory)
        & np.isfinite(h_cross_no_memory)
    )
    t_raw = t_raw[finite]
    h_plus = h_plus[finite]
    h_cross = h_cross[finite]
    h_plus_no_memory = h_plus_no_memory[finite]
    h_cross_no_memory = h_cross_no_memory[finite]
    if len(t_raw) < 4:
        raise ValueError("waveform is too short")

    memory_start_offsets = {
        "h_plus": float(h_plus[0] - h_plus_no_memory[0]),
        "h_cross": float(h_cross[0] - h_cross_no_memory[0]),
    }
    h_plus = h_plus - memory_start_offsets["h_plus"]
    h_cross = h_cross - memory_start_offsets["h_cross"]

    t = t_raw - t_raw[0]
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1.0e-10, atol=max(1.0e-12, abs(dt) * 1.0e-12)):
        raise ValueError("waveform time grid must be uniform")
    metadata = parse_metadata(data)
    metadata.update(
        {
            "waveform_npz": str(path),
            "hp_key": hp_key,
            "hc_key": hc_key,
            "h_plus_no_memory_key": "h_plus_osc",
            "h_cross_no_memory_key": "h_cross_osc",
            "memory_start_alignment": {
                "rule": "total polarizations are shifted by the initial total-minus-no-memory offset so memory starts at zero",
                "offset_subtracted": memory_start_offsets,
                "first_sample_after_alignment": {
                    "h_plus_total_minus_no_memory": float(h_plus[0] - h_plus_no_memory[0]),
                    "h_cross_total_minus_no_memory": float(h_cross[0] - h_cross_no_memory[0]),
                },
            },
            "input_start_time_s": float(t_raw[0]),
            "shifted_time_start_s": 0.0,
            "samples_used": int(len(t)),
            "dt_s": dt,
        }
    )
    return t, h_plus, h_cross, h_plus_no_memory, h_cross_no_memory, metadata


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


def interpolate_series(t_source: np.ndarray, values: np.ndarray, t_query: float) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(len(t_source), -1)
    out = np.empty(flat.shape[1], dtype=float)
    for i in range(flat.shape[1]):
        out[i] = np.interp(float(t_query), t_source, flat[:, i])
    return out.reshape(values.shape[1:])


def source_gate(n: int, edge: int, alpha: float) -> np.ndarray:
    if 2 * edge >= n:
        raise ValueError("t_buffer removes all samples; reduce --t-buffer or use a longer waveform")
    gate = np.zeros(n, dtype=float)
    gate[edge:-edge] = tukey_window(n - 2 * edge, alpha)
    return gate


def spectrum_from_time_derivative(
    values: np.ndarray,
    dt: float,
    *,
    derivative_window: np.ndarray | None = None,
) -> np.ndarray:
    """Recover a one-sided Fourier transform from FFT(dh/dt)/(i 2 pi f).

    For displacement-memory waveforms the taper must be applied to dh/dt, not to
    h itself; tapering h would add artificial endpoint derivatives.
    """
    values = np.asarray(values, dtype=float)
    freqs = np.fft.rfftfreq(len(values), d=dt)
    derivative = np.gradient(values, dt, edge_order=2)
    if derivative_window is not None:
        derivative = derivative * np.asarray(derivative_window, dtype=float)
    derivative_f = dt * np.fft.rfft(derivative)
    spectrum = np.zeros_like(derivative_f)
    omega = 2.0 * np.pi * freqs
    spectrum[1:] = derivative_f[1:] / (1j * omega[1:])
    return spectrum


def spectrum_from_windowed_signal(values: np.ndarray, dt: float, window: np.ndarray) -> np.ndarray:
    """Return the FFT of a windowed time-domain detector observable."""

    return dt * np.fft.rfft(np.asarray(window, dtype=float) * np.asarray(values, dtype=float))


def smooth_highpass(freqs: np.ndarray, f_stop: float, f_pass: float) -> np.ndarray:
    """Raised-cosine high-pass gate that is exactly one above ``f_pass``."""

    if f_stop <= 0.0 or f_pass <= f_stop:
        raise ValueError("high-pass frequencies must satisfy 0 < f_stop < f_pass")
    freqs = np.asarray(freqs, dtype=float)
    gate = np.ones_like(freqs)
    gate[freqs <= f_stop] = 0.0
    transition = (freqs > f_stop) & (freqs < f_pass)
    x = (freqs[transition] - f_stop) / (f_pass - f_stop)
    gate[transition] = 0.5 * (1.0 - np.cos(np.pi * x))
    return gate


def power_low_frequency_gate(freqs: np.ndarray, f_pass: float, power: float) -> np.ndarray:
    """Continuous leakage-suppression gate, unity at and above ``f_pass``."""

    if f_pass <= 0.0 or power <= 0.0:
        raise ValueError("f_pass and power must be positive")
    freqs = np.asarray(freqs, dtype=float)
    gate = np.ones_like(freqs)
    low = freqs < f_pass
    gate[low] = np.power(np.maximum(freqs[low], 0.0) / f_pass, power)
    return gate


def default_reference_time(t: np.ndarray, h_plus: np.ndarray, h_cross: np.ndarray, metadata: dict[str, object]) -> float:
    params_peak = metadata.get("peak_time_s")
    start = metadata.get("input_start_time_s", 0.0)
    if params_peak is not None:
        candidate = float(params_peak) - float(start)
        if t[0] <= candidate <= t[-1]:
            return candidate
    return float(t[int(np.argmax(np.hypot(h_plus, h_cross)))])


def compute_td_ae(
    *,
    t: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    orbits,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    response = FastLISAResponseTDI(
        orbits=orbits,
        order=args.order,
        tdi="2nd generation",
        tdi_chan="AE",
        force_backend=args.response_backend,
        t_buffer=args.t_buffer,
        trim_garbage=False,
        cache_response=False,
    )
    result = response.compute(t, h_plus, h_cross, lam=args.lam, beta=args.beta, t0=0.0)
    out = result.as_numpy()
    return (
        np.asarray(out["t"], dtype=float),
        np.asarray(out["A"], dtype=float),
        np.asarray(out["E"], dtype=float),
        dict(result.metadata),
    )


def rel_l2(model: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((model - ref)[mask]) / max(np.linalg.norm(ref[mask]), 1.0e-300))


def overlap_abs(model: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    numerator = abs(np.vdot(ref[mask], model[mask]))
    denominator = np.linalg.norm(ref[mask]) * np.linalg.norm(model[mask])
    return float(numerator / max(denominator, 1.0e-300))


def loglog_slope(freqs: np.ndarray, values: np.ndarray, fmin: float, fmax: float) -> float | None:
    amp = np.abs(values)
    mask = (freqs >= fmin) & (freqs <= fmax) & np.isfinite(amp) & (amp > 0.0)
    if np.count_nonzero(mask) < 3:
        return None
    slope, _intercept = np.polyfit(np.log10(freqs[mask]), np.log10(amp[mask]), 1)
    return float(slope)


def compare_spectra(
    *,
    t: np.ndarray,
    h_plus_td: np.ndarray,
    h_cross_td: np.ndarray,
    h_plus_no_memory_td: np.ndarray,
    h_cross_no_memory_td: np.ndarray,
    realistic: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]],
    realistic_no_memory: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]],
    static: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]],
    static_no_memory: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]],
    static_positions_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    t_static, a_static, e_static, meta_static = static
    t_static_no_memory, a_static_no_memory, e_static_no_memory, _meta_static_no_memory = static_no_memory
    t_realistic, a_realistic, e_realistic, _meta_realistic = realistic
    t_no_memory, a_no_memory, e_no_memory, _meta_no_memory = realistic_no_memory
    n = min(len(t), len(t_static), len(t_static_no_memory), len(t_realistic), len(t_no_memory))
    t_use = t[:n]
    h_plus_td = h_plus_td[:n]
    h_cross_td = h_cross_td[:n]
    h_plus_no_memory_td = h_plus_no_memory_td[:n]
    h_cross_no_memory_td = h_cross_no_memory_td[:n]
    t_static = t_static[:n]
    a_static = a_static[:n]
    e_static = e_static[:n]
    t_static_no_memory = t_static_no_memory[:n]
    a_static_no_memory = a_static_no_memory[:n]
    e_static_no_memory = e_static_no_memory[:n]
    t_realistic = t_realistic[:n]
    a_realistic = a_realistic[:n]
    e_realistic = e_realistic[:n]
    t_no_memory = t_no_memory[:n]
    a_no_memory = a_no_memory[:n]
    e_no_memory = e_no_memory[:n]
    if (
        not np.allclose(t_static, t_use)
        or not np.allclose(t_static_no_memory, t_use)
        or not np.allclose(t_realistic, t_use)
        or not np.allclose(t_no_memory, t_use)
    ):
        raise ValueError("untrimmed TDI outputs must share the input time grid")

    dt = float(np.median(np.diff(t_use)))
    freqs = np.fft.rfftfreq(n, d=dt)
    hp_f = spectrum_from_time_derivative(h_plus_td, dt)
    hc_f = spectrum_from_time_derivative(h_cross_td, dt)
    hp_no_memory_f = spectrum_from_time_derivative(h_plus_no_memory_td, dt)
    hc_no_memory_f = spectrum_from_time_derivative(h_cross_no_memory_td, dt)
    hp_memory_only_f = spectrum_from_time_derivative(h_plus_td - h_plus_no_memory_td, dt)
    hc_memory_only_f = spectrum_from_time_derivative(h_cross_td - h_cross_no_memory_td, dt)
    fd_response = StaticTaijiFDResponse(positions_m=static_positions_m)
    a_fd_full, e_fd_full = fd_response.ae(
        freqs,
        hp_f,
        hc_f,
        lam=args.lam,
        beta=args.beta,
        tdi="2nd generation",
    )
    a_fd_no_memory_full, e_fd_no_memory_full = fd_response.ae(
        freqs,
        hp_no_memory_f,
        hc_no_memory_f,
        lam=args.lam,
        beta=args.beta,
        tdi="2nd generation",
    )
    a_fd_time = np.fft.irfft(a_fd_full / dt, n=n)
    e_fd_time = np.fft.irfft(e_fd_full / dt, n=n)
    a_fd_no_memory_time = np.fft.irfft(a_fd_no_memory_full / dt, n=n)
    e_fd_no_memory_time = np.fft.irfft(e_fd_no_memory_full / dt, n=n)

    guard = int(round(args.comparison_guard_s / dt)) if args.comparison_guard_s is not None else int(meta_static["tdi_start_ind"])
    if 2 * guard >= n:
        raise ValueError("comparison guard removes all samples; reduce --comparison-guard-s or --t-buffer")
    comparison_window = np.zeros(n, dtype=float)
    comparison_window[guard:-guard] = tukey_window(n - 2 * guard, args.fft_window_alpha)
    a_static_f = spectrum_from_windowed_signal(a_static, dt, comparison_window)
    e_static_f = spectrum_from_windowed_signal(e_static, dt, comparison_window)
    a_static_no_memory_f = spectrum_from_windowed_signal(a_static_no_memory, dt, comparison_window)
    e_static_no_memory_f = spectrum_from_windowed_signal(e_static_no_memory, dt, comparison_window)
    a_realistic_f = spectrum_from_windowed_signal(a_realistic, dt, comparison_window)
    e_realistic_f = spectrum_from_windowed_signal(e_realistic, dt, comparison_window)
    a_no_memory_f = spectrum_from_windowed_signal(a_no_memory, dt, comparison_window)
    e_no_memory_f = spectrum_from_windowed_signal(e_no_memory, dt, comparison_window)
    a_memory_only_f = spectrum_from_windowed_signal(a_realistic - a_no_memory, dt, comparison_window)
    e_memory_only_f = spectrum_from_windowed_signal(e_realistic - e_no_memory, dt, comparison_window)
    a_fd = spectrum_from_windowed_signal(a_fd_time, dt, comparison_window)
    e_fd = spectrum_from_windowed_signal(e_fd_time, dt, comparison_window)
    a_fd_no_memory = spectrum_from_windowed_signal(a_fd_no_memory_time, dt, comparison_window)
    e_fd_no_memory = spectrum_from_windowed_signal(e_fd_no_memory_time, dt, comparison_window)

    cleanup_gate = np.ones_like(freqs)
    cleanup_summary: dict[str, object] = {
        "enabled": bool(args.clean_oscillatory_leakage),
    }
    if args.clean_oscillatory_leakage:
        f_pass = float(args.oscillatory_cleanup_pass_freq)
        cleanup_gate = power_low_frequency_gate(freqs, f_pass, float(args.oscillatory_cleanup_power))
        cleanup_summary.update(
            {
                "f_pass_hz": f_pass,
                "power": float(args.oscillatory_cleanup_power),
                "rule": "applied only to no-memory oscillatory A/E display spectra; gate=(f/f_pass)^power below f_pass and unity at/above f_pass",
            }
        )
    a_static_display = (a_static_f - a_static_no_memory_f) + cleanup_gate * a_static_no_memory_f
    e_static_display = (e_static_f - e_static_no_memory_f) + cleanup_gate * e_static_no_memory_f
    a_realistic_display = a_memory_only_f + cleanup_gate * a_no_memory_f
    e_realistic_display = e_memory_only_f + cleanup_gate * e_no_memory_f
    a_no_memory_display = cleanup_gate * a_no_memory_f
    e_no_memory_display = cleanup_gate * e_no_memory_f
    a_fd_display = (a_fd - a_fd_no_memory) + cleanup_gate * a_fd_no_memory
    e_fd_display = (e_fd - e_fd_no_memory) + cleanup_gate * e_fd_no_memory

    amp = np.maximum(np.abs(a_realistic_f), np.abs(e_realistic_f))
    floor = args.relative_amp_floor * float(np.max(amp))
    mask = (freqs >= args.compare_fmin) & (freqs <= args.compare_fmax) & (amp > floor)
    mask &= np.isfinite(a_fd) & np.isfinite(e_fd) & np.isfinite(a_realistic_f) & np.isfinite(e_realistic_f)
    if not np.any(mask):
        raise ValueError("comparison mask is empty")

    summary = {
        "n_fft": int(n),
        "dt_s": dt,
        "df_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
        "mask_bins": int(np.count_nonzero(mask)),
        "compare_fmin_hz": float(args.compare_fmin),
        "compare_fmax_hz": float(args.compare_fmax),
        "source_gate_tukey_alpha": None if args.source_gate_alpha < 0.0 else float(args.source_gate_alpha),
        "memory_spectrum_method": "FFT(time derivative)/(i 2 pi f), f=0 set to 0",
        "comparison_guard_s": float(guard * dt),
        "comparison_guard_samples": int(guard),
        "comparison_window_tukey_alpha": float(args.fft_window_alpha),
        "tdi_spectrum_method": "FFT(windowed time-domain A/E)",
        "source_memory_spectrum_method": "FFT(time derivative)/(i 2 pi f), f=0 set to 0",
        "oscillatory_leakage_cleanup": cleanup_summary,
        "fd_vs_td_reference": "realistic Taiji orbit TD",
        "realistic_A_fd_vs_td_relative_l2": rel_l2(a_fd, a_realistic_f, mask),
        "realistic_E_fd_vs_td_relative_l2": rel_l2(e_fd, e_realistic_f, mask),
        "realistic_A_fd_vs_td_overlap_abs": overlap_abs(a_fd, a_realistic_f, mask),
        "realistic_E_fd_vs_td_overlap_abs": overlap_abs(e_fd, e_realistic_f, mask),
        "static_A_fd_vs_td_relative_l2": rel_l2(a_fd, a_static_f, mask),
        "static_E_fd_vs_td_relative_l2": rel_l2(e_fd, e_static_f, mask),
        "static_A_fd_vs_td_overlap_abs": overlap_abs(a_fd, a_static_f, mask),
        "static_E_fd_vs_td_overlap_abs": overlap_abs(e_fd, e_static_f, mask),
        "realistic_vs_static_A_relative_l2": rel_l2(a_realistic_f, a_static_f, mask),
        "realistic_vs_static_E_relative_l2": rel_l2(e_realistic_f, e_static_f, mask),
        "memory_only_A_low_frequency_slope": loglog_slope(freqs, a_memory_only_f, 2.0e-5, 1.5e-4),
        "memory_only_E_low_frequency_slope": loglog_slope(freqs, e_memory_only_f, 2.0e-5, 1.5e-4),
        "source_memory_only_h_plus_low_frequency_slope": loglog_slope(freqs, hp_memory_only_f, 2.0e-5, 1.5e-4),
        "source_memory_only_h_cross_low_frequency_slope": loglog_slope(freqs, hc_memory_only_f, 2.0e-5, 1.5e-4),
        "memory_only_slope_band_hz": [2.0e-5, 1.5e-4],
    }
    arrays = {
        "freqs": freqs,
        "mask": mask,
        "A_realistic_td": a_realistic_f,
        "E_realistic_td": e_realistic_f,
        "A_realistic_td_display": a_realistic_display,
        "E_realistic_td_display": e_realistic_display,
        "A_realistic_no_memory_td": a_no_memory_f,
        "E_realistic_no_memory_td": e_no_memory_f,
        "A_realistic_no_memory_td_display": a_no_memory_display,
        "E_realistic_no_memory_td_display": e_no_memory_display,
        "A_realistic_memory_only_td": a_memory_only_f,
        "E_realistic_memory_only_td": e_memory_only_f,
        "A_static_td": a_static_f,
        "E_static_td": e_static_f,
        "A_static_td_display": a_static_display,
        "E_static_td_display": e_static_display,
        "A_static_no_memory_td": a_static_no_memory_f,
        "E_static_no_memory_td": e_static_no_memory_f,
        "A_static_fd": a_fd,
        "E_static_fd": e_fd,
        "A_static_fd_display": a_fd_display,
        "E_static_fd_display": e_fd_display,
        "A_static_fd_no_memory": a_fd_no_memory,
        "E_static_fd_no_memory": e_fd_no_memory,
        "A_static_fd_full": a_fd_full,
        "E_static_fd_full": e_fd_full,
        "A_static_fd_no_memory_full": a_fd_no_memory_full,
        "E_static_fd_no_memory_full": e_fd_no_memory_full,
        "A_static_fd_time": a_fd_time,
        "E_static_fd_time": e_fd_time,
        "A_static_fd_no_memory_time": a_fd_no_memory_time,
        "E_static_fd_no_memory_time": e_fd_no_memory_time,
        "hp_f": hp_f,
        "hc_f": hc_f,
        "hp_no_memory_f": hp_no_memory_f,
        "hc_no_memory_f": hc_no_memory_f,
        "hp_memory_only_f": hp_memory_only_f,
        "hc_memory_only_f": hc_memory_only_f,
        "comparison_window": comparison_window,
        "oscillatory_cleanup_gate": cleanup_gate,
    }
    return arrays, summary


def plot_outputs(
    *,
    t: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    h_plus_no_memory: np.ndarray,
    h_cross_no_memory: np.ndarray,
    spectra: dict[str, np.ndarray],
    summary: dict[str, object],
    figure_path: Path,
    oscillatory_start_freq_hz: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = spectra["freqs"]
    mask = spectra["mask"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(t / 3600.0, h_plus, lw=0.9, label=r"$h_+$")
    axes[0, 0].plot(t / 3600.0, h_cross, lw=0.9, label=r"$h_\times$")
    axes[0, 0].plot(t / 3600.0, h_plus_no_memory, "k--", lw=1.0, label=r"$h_+^{\rm no\,mem}$")
    axes[0, 0].plot(t / 3600.0, h_cross_no_memory, ":", color="0.15", lw=1.0, label=r"$h_\times^{\rm no\,mem}$")
    axes[0, 0].set_xlabel(r"$t$ [h]")
    axes[0, 0].set_ylabel(r"$h(t)$")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")

    eps = 1.0e-300
    for ax, channel in ((axes[0, 1], "A"), (axes[1, 0], "E")):
        realistic = spectra[f"{channel}_realistic_td"]
        realistic_cleaned = spectra[f"{channel}_realistic_td_display"]
        no_memory = spectra[f"{channel}_realistic_no_memory_td"]
        memory_only = spectra[f"{channel}_realistic_memory_only_td"]
        static = spectra[f"{channel}_static_td"]
        fd = spectra[f"{channel}_static_fd"]
        freq_scale = 1.0 / (freqs[1:] ** 2)
        ax.loglog(freqs[1:], freq_scale * np.abs(realistic[1:]), lw=1.0, label="realistic Taiji orbit")
        if bool(summary.get("oscillatory_leakage_cleanup", {}).get("enabled", False)):
            ax.loglog(
                freqs[1:],
                freq_scale * np.abs(realistic_cleaned[1:]),
                "-.",
                color="0.05",
                lw=1.0,
                label="realistic Taiji orbit (leakage suppressed)",
            )
        ax.loglog(freqs[1:], freq_scale * np.abs(no_memory[1:]), "k--", lw=1.0, label="realistic Taiji orbit (no memory)")
        ax.loglog(
            freqs[1:],
            freq_scale * np.abs(memory_only[1:]),
            ":",
            color="0.25",
            lw=1.0,
            label="realistic Taiji orbit (memory only)",
        )
        ax.loglog(freqs[1:], freq_scale * np.abs(static[1:]), lw=1.0, label="static equal-arm")
        ax.loglog(freqs[1:], freq_scale * np.abs(fd[1:]), "--", lw=1.0, label="analytic response")
        if oscillatory_start_freq_hz is not None and oscillatory_start_freq_hz > 0.0:
            ax.axvline(
                oscillatory_start_freq_hz,
                color="0.35",
                ls="-.",
                lw=0.9,
                label="f22_start",
            )
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_ylabel(rf"$|\tilde{{{channel}}}(f)|/f^2$")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    rel_a = np.abs(spectra["A_static_fd"] - spectra["A_realistic_td"]) / np.maximum(
        np.abs(spectra["A_realistic_td"]), eps
    )
    rel_e = np.abs(spectra["E_static_fd"] - spectra["E_realistic_td"]) / np.maximum(
        np.abs(spectra["E_realistic_td"]), eps
    )
    axes[1, 1].loglog(freqs[mask], rel_a[mask], lw=1.0, label=r"$\tilde{A}$")
    axes[1, 1].loglog(freqs[mask], rel_e[mask], lw=1.0, label=r"$\tilde{E}$")
    axes[1, 1].set_xlabel(r"$f$ [Hz]")
    axes[1, 1].set_ylabel(r"$|\tilde{U}_{\rm ana}-\tilde{U}_{\rm TD}|/|\tilde{U}_{\rm TD}|$")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="best")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    tic = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t, h_plus, h_cross, h_plus_no_memory, h_cross_no_memory, waveform_metadata = load_waveform(args.waveform_npz)
    dt = float(np.median(np.diff(t)))
    edge = int(args.t_buffer / dt)
    if args.source_gate_alpha < 0.0:
        gate = np.ones(len(t), dtype=float)
    else:
        gate = source_gate(len(t), edge, args.source_gate_alpha)
    h_plus_td = h_plus * gate
    h_cross_td = h_cross * gate
    h_plus_no_memory_td = h_plus_no_memory * gate
    h_cross_no_memory_td = h_cross_no_memory * gate
    reference_time = args.reference_time_s
    if reference_time is None:
        reference_time = default_reference_time(t, h_plus, h_cross, waveform_metadata)
    reference_time = float(reference_time)
    params = waveform_metadata.get("params", {})
    if args.oscillatory_start_freq is None:
        args.oscillatory_start_freq = float(params.get("f22_start", args.compare_fmin))
    if args.oscillatory_cleanup_pass_freq is None:
        args.oscillatory_cleanup_pass_freq = float(args.oscillatory_start_freq)

    orbit_duration = float(t[-1] + dt + args.orbit_margin_s)
    orbit_spec = {"base": "taiji-accurate", "orbit_dt": args.orbit_dt}
    if args.taiji_orbit_dir is not None:
        orbit_spec["orbit_dir"] = str(args.taiji_orbit_dir)
    raw_realistic = make_orbits_from_spec(
        orbit_spec,
        duration=orbit_duration,
        force_backend=args.response_backend,
    )
    realistic_standard = make_standard_convention_orbits(raw_realistic, force_backend=args.response_backend)
    reference_positions = interpolate_series(realistic_standard.t_base, realistic_standard.x_base, reference_time)
    static_orbits, static_match = make_static_equal_arm_orbits_from_reference(
        reference_positions,
        duration_s=orbit_duration,
        reference_time_s=reference_time,
        center_at_reference=True,
        force_backend=args.response_backend,
    )

    realistic_td = compute_td_ae(t=t, h_plus=h_plus_td, h_cross=h_cross_td, orbits=realistic_standard, args=args)
    realistic_no_memory_td = compute_td_ae(
        t=t,
        h_plus=h_plus_no_memory_td,
        h_cross=h_cross_no_memory_td,
        orbits=realistic_standard,
        args=args,
    )
    static_td = compute_td_ae(t=t, h_plus=h_plus_td, h_cross=h_cross_td, orbits=static_orbits, args=args)
    static_no_memory_td = compute_td_ae(
        t=t,
        h_plus=h_plus_no_memory_td,
        h_cross=h_cross_no_memory_td,
        orbits=static_orbits,
        args=args,
    )
    spectra, comparison = compare_spectra(
        t=t,
        h_plus_td=h_plus_td,
        h_cross_td=h_cross_td,
        h_plus_no_memory_td=h_plus_no_memory_td,
        h_cross_no_memory_td=h_cross_no_memory_td,
        realistic=realistic_td,
        realistic_no_memory=realistic_no_memory_td,
        static=static_td,
        static_no_memory=static_no_memory_td,
        static_positions_m=static_match.positions_m,
        args=args,
    )

    output_npz = args.output_dir / "taiji_static_tdi2_memory_demo.npz"
    figure_path = args.output_dir / "taiji_static_tdi2_memory_demo.png"
    summary_path = args.output_dir / "summary.json"
    static_meta = asdict(static_match)
    static_meta = {
        key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
        for key, value in static_meta.items()
    }
    summary = {
        "waveform": waveform_metadata,
        "convention": {
            "orbit_file_to_analysis": "raw Taiji orbit relabeled to the standard LISA/TDI convention before TDI",
            "raw_to_standard_spacecraft": "raw SC1=standard SC2, raw SC2=standard SC1, raw SC3=standard SC3",
            "raw_to_standard_ltt_indices_for_12_23_31_13_32_21": [5, 3, 4, 1, 2, 0],
        },
        "source": {
            "lam_rad": float(args.lam),
            "beta_rad": float(args.beta),
            "reference_time_s": reference_time,
            "source_gate_edge_samples": 0 if args.source_gate_alpha < 0.0 else int(edge),
            "source_gate_tukey_alpha": None if args.source_gate_alpha < 0.0 else float(args.source_gate_alpha),
            "no_memory_reference": "realistic Taiji response to h_plus_osc and h_cross_osc",
            "frequency_domain_memory_treatment": "spectra are recovered from FFT(time derivative)/(i 2 pi f)",
            "plotted_detector_spectra": "raw normally windowed A/E spectra plus an explicit leakage-suppressed display spectrum, all divided by f^2",
            "memory_max_abs": {
                "h_plus": float(np.max(np.abs(h_plus - h_plus_no_memory))),
                "h_cross": float(np.max(np.abs(h_cross - h_cross_no_memory))),
            },
            "f22_start_hz": float(args.oscillatory_start_freq),
            "oscillatory_start_freq_hz": float(args.oscillatory_start_freq),
            "oscillatory_cleanup_pass_freq_hz": float(args.oscillatory_cleanup_pass_freq),
        },
        "response": {
            "tdi": "2nd generation",
            "tdi_chan": "AE",
            "order": int(args.order),
            "t_buffer_s": float(args.t_buffer),
            "response_backend": args.response_backend,
            "orbit_dt_s": float(args.orbit_dt),
            "static_orbit_center": "SSB reference center",
        },
        "static_equal_arm_orbit_match": static_meta,
        "comparison": comparison,
        "outputs": {
            "npz": str(output_npz),
            "figure": str(figure_path),
            "summary": str(summary_path),
        },
        "elapsed_s": float(time.perf_counter() - tic),
    }

    np.savez(
        output_npz,
        t_seconds=t,
        h_plus=h_plus,
        h_cross=h_cross,
        h_plus_no_memory=h_plus_no_memory,
        h_cross_no_memory=h_cross_no_memory,
        h_plus_td=h_plus_td,
        h_cross_td=h_cross_td,
        h_plus_no_memory_td=h_plus_no_memory_td,
        h_cross_no_memory_td=h_cross_no_memory_td,
        source_gate=gate,
        realistic_t_seconds=realistic_td[0],
        realistic_A=realistic_td[1],
        realistic_E=realistic_td[2],
        realistic_no_memory_t_seconds=realistic_no_memory_td[0],
        realistic_no_memory_A=realistic_no_memory_td[1],
        realistic_no_memory_E=realistic_no_memory_td[2],
        static_t_seconds=static_td[0],
        static_A=static_td[1],
        static_E=static_td[2],
        static_no_memory_t_seconds=static_no_memory_td[0],
        static_no_memory_A=static_no_memory_td[1],
        static_no_memory_E=static_no_memory_td[2],
        static_positions_m=static_match.positions_m,
        reference_positions_m=reference_positions,
        **spectra,
        summary_json=np.asarray(json.dumps(summary, separators=(",", ":"), default=str)),
    )
    plot_outputs(
        t=t,
        h_plus=h_plus,
        h_cross=h_cross,
        h_plus_no_memory=h_plus_no_memory,
        h_cross_no_memory=h_cross_no_memory,
        spectra=spectra,
        summary=comparison,
        figure_path=figure_path,
        oscillatory_start_freq_hz=args.oscillatory_start_freq,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-npz", type=Path, default=DEFAULT_WAVEFORM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--response-backend", choices=["cpu", "cuda12x"], default="cpu")
    parser.add_argument(
        "--taiji-orbit-dir",
        type=Path,
        default=None,
        help="Path to downloaded Triangle-Simulator/OrbitData/MicroSateOrbitEclipticTCB.",
    )
    parser.add_argument("--orbit-dt", type=float, default=600.0)
    parser.add_argument("--orbit-margin-s", type=float, default=1200.0)
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument("--t-buffer", type=float, default=4096.0)
    parser.add_argument(
        "--source-gate-alpha",
        type=float,
        default=-1.0,
        help="Disable source gating when negative; otherwise use this Tukey alpha after t_buffer edges.",
    )
    parser.add_argument("--fft-window-alpha", type=float, default=0.2)
    parser.add_argument("--comparison-guard-s", type=float, default=None)
    parser.add_argument("--compare-fmin", type=float, default=2.0e-4)
    parser.add_argument("--compare-fmax", type=float, default=1.5e-2)
    parser.add_argument("--relative-amp-floor", type=float, default=1.0e-7)
    parser.add_argument(
        "--clean-oscillatory-leakage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For display spectra, suppress no-memory finite-window leakage below the physical oscillatory start frequency.",
    )
    parser.add_argument("--oscillatory-start-freq", type=float, default=None)
    parser.add_argument("--oscillatory-cleanup-pass-freq", type=float, default=None)
    parser.add_argument("--oscillatory-cleanup-power", type=float, default=8.0)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--reference-time-s", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
