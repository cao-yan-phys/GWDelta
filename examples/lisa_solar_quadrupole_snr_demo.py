"""One-year ESA-LISA response and SNR for a solar mass quadrupole."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from weak_field_demo_common import (
    DAY_S,
    REPO_ROOT,
    SOLAR_EQUATOR_ASCENDING_NODE_DEG,
    SOLAR_EQUATOR_INCLINATION_DEG,
    SOLAR_G1_M2_DEFAULT_DISK_VELOCITY_M_S,
    SOLAR_G1_M2_FREQUENCY_HZ,
    as_numpy,
    build_time_grids,
    calibrated_solar_g1_m2_quadrupole,
    compute_ae_snr,
    configure_cuda_if_needed,
    configure_latex_roman,
    display_scale,
    make_esa_lisa_orbits,
    make_response_engines,
    tdi_channels_numpy,
    write_summary,
)
from gwdelta import RetardedQuadrupoleMode, select_array_backend


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lisa_solar_quadrupole_snr"
README_FIGURE_NAME = "lisa_solar_quadrupole_snr_demo.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=30.0)
    parser.add_argument("--orbit-dt", type=float, default=600.0)
    parser.add_argument(
        "--response-backend", choices=["cpu", "cuda12x"], default="cuda12x"
    )
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--tdi-order", type=int, default=15)
    parser.add_argument("--t-buffer", type=float, default=10000.0)
    parser.add_argument(
        "--disk-velocity-m-s",
        type=float,
        default=SOLAR_G1_M2_DEFAULT_DISK_VELOCITY_M_S,
        help="whole-disk line-of-sight velocity amplitude of the solar mode",
    )
    parser.add_argument("--solar-phase", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument(
        "--publish-figure",
        action="store_true",
        help="also write the reproducible README figure under docs/figures",
    )
    return parser.parse_args()


def make_figure(
    path: Path,
    *,
    channels: dict[str, np.ndarray],
    snr: dict[str, object],
    solar_frequency_hz: float,
) -> None:
    import matplotlib.pyplot as plt

    configure_latex_roman()
    colors = {"A": "#D55E00", "E": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)

    time_s = channels["t"]
    mask = time_s <= DAY_S
    scale, exponent = display_scale(channels["A"][mask], channels["E"][mask])
    for channel in ("A", "E"):
        axes[0].plot(
            time_s[mask] / DAY_S,
            channels[channel][mask] / scale,
            color=colors[channel],
            lw=0.9,
            label=rf"${channel}$",
        )
    axes[0].set_xlabel(r"$t\,[{\rm day}]$")
    axes[0].set_ylabel(rf"$U(t)\ [10^{{{exponent}}}]$")
    axes[0].legend(frameon=False, ncol=2, loc="lower right")

    frequency_hz = np.asarray(snr["frequency_hz"])
    positive = frequency_hz > 0.0
    for channel in ("A", "E"):
        axes[1].loglog(
            frequency_hz[positive],
            np.abs(snr["spectra"][channel][positive]),
            color=colors[channel],
            lw=0.9,
            label=rf"$|\widetilde{{{channel}}}(f)|$",
        )
    axes[1].axvline(
        solar_frequency_hz,
        color="0.2",
        ls="--",
        lw=0.9,
        label=r"$f_Q$",
    )
    axes[1].set_xlim(
        max(frequency_hz[1], solar_frequency_hz / 20.0),
        solar_frequency_hz * 20.0,
    )
    axes[1].set_xlabel(r"$f\,[{\rm Hz}]$")
    axes[1].set_ylabel(r"$|\widetilde A(f)|,\ |\widetilde E(f)|\,[{\rm s}]$")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.tick_params(direction="in", which="both", top=True, right=True)
        axis.grid(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    configure_cuda_if_needed(args.response_backend)
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_time, response_time, analysis_samples = build_time_grids(
        args.years, args.dt, args.t_buffer
    )

    tic = time.perf_counter()
    orbits = make_esa_lisa_orbits(motion_time[-1], args.orbit_dt, args.response_backend)
    orbit_setup_s = time.perf_counter() - tic
    quadrupole_amplitude, solar_calibration = calibrated_solar_g1_m2_quadrupole(
        args.disk_velocity_m_s, args.solar_phase
    )
    source = RetardedQuadrupoleMode(
        angular_frequency=2.0 * np.pi * SOLAR_G1_M2_FREQUENCY_HZ,
        quadrupole_amplitude_kg_m2=quadrupole_amplitude,
        origin_m=(0.0, 0.0, 0.0),
    )
    t_base = as_numpy(orbits.t_base).astype(float, copy=False)
    x_base = as_numpy(orbits.x_base).astype(float, copy=False)
    background_position = CubicSpline(t_base, x_base, axis=0)(motion_time)
    array_backend = select_array_backend(
        force="cupy" if args.response_backend == "cuda12x" else "cpu"
    )
    tic = time.perf_counter()
    motion = source.steady_state_test_mass_motion(
        array_backend.asarray(motion_time),
        array_backend.asarray(background_position),
    )
    worldline_setup_s = time.perf_counter() - tic
    link_engine, tdi_engine = make_response_engines(
        orbits,
        response_backend=args.response_backend,
        quadrature_order=args.quadrature_order,
        chunk_size=args.chunk_size,
        tdi_order=args.tdi_order,
        t_buffer_s=args.t_buffer,
    )

    tic = time.perf_counter()
    links = link_engine.compute(
        response_time,
        source,
        delta_velocity_m_s=motion,
    )
    tdi_result = tdi_engine.compute_links(response_time, links.total)
    channels = tdi_channels_numpy(tdi_result)
    response_s = time.perf_counter() - tic
    if len(channels["t"]) != analysis_samples:
        raise RuntimeError(
            "TDI trimming did not produce the requested observation length"
        )
    if not all(np.all(np.isfinite(channels[name])) for name in ("A", "E")):
        raise RuntimeError("the solar TDI response contains non-finite values")

    snr = compute_ae_snr(channels["t"], channels)
    figure_path = output_dir / README_FIGURE_NAME
    make_figure(
        figure_path,
        channels=channels,
        snr=snr,
        solar_frequency_hz=SOLAR_G1_M2_FREQUENCY_HZ,
    )
    published_figure = None
    if args.publish_figure:
        published_figure = REPO_ROOT / "docs" / "figures" / README_FIGURE_NAME
        make_figure(
            published_figure,
            channels=channels,
            snr=snr,
            solar_frequency_hz=SOLAR_G1_M2_FREQUENCY_HZ,
        )

    npz_path = output_dir / "lisa_solar_quadrupole_snr_demo.npz"
    if args.save_npz:
        np.savez_compressed(
            npz_path,
            t_seconds=channels["t"],
            A=channels["A"],
            E=channels["E"],
            frequency_hz=snr["frequency_hz"],
            A_fft=snr["spectra"]["A"],
            E_fft=snr["spectra"]["E"],
            lisa_S_AE=snr["psd"],
        )

    summary = {
        "example": "ESA LISA solar quadrupole SNR",
        "orbit": {
            "base": "esa",
            "source": "ESAOrbits from lisatools",
            "coordinate_transform": "none",
            "orbit_dt_s": float(args.orbit_dt),
            "backend": getattr(orbits.backend, "name", None),
            "link_order": list(orbits.LINKS),
        },
        "observation": {
            "requested_years": float(args.years),
            "samples": int(len(channels["t"])),
            "dt_s": float(args.dt),
            "duration_s": float(len(channels["t"]) * args.dt),
            "start_time_s": float(channels["t"][0]),
            "stop_time_s": float(channels["t"][-1]),
            "tdi_generation": "second",
            "tdi_channels": "AE",
        },
        "solar_quadrupole": {
            "mode": "solar ell=2, m=2, n=-1 g mode",
            "frequency_hz": SOLAR_G1_M2_FREQUENCY_HZ,
            "quadrupole_frobenius_norm_kg_m2": float(
                np.linalg.norm(quadrupole_amplitude)
            ),
            "phase_rad": float(args.solar_phase),
            "origin_m": [0.0, 0.0, 0.0],
            "solar_equator_inclination_deg": SOLAR_EQUATOR_INCLINATION_DEG,
            "solar_equator_ascending_node_deg": SOLAR_EQUATOR_ASCENDING_NODE_DEG,
            "physical_calibration": solar_calibration,
            "response_scope": (
                "photon-propagation response plus leading nonrelativistic "
                "steady-state endpoint velocity"
            ),
            "endpoint_motion": {
                "method": "monochromatic forced motion on the ESA background orbit",
                "max_abs_acceleration_m_s2": float(
                    np.max(np.abs(as_numpy(motion.acceleration_m_s2)))
                ),
                "max_abs_delta_velocity_m_s": float(
                    np.max(np.abs(as_numpy(motion.delta_velocity_m_s)))
                ),
                "max_abs_delta_position_m": float(
                    np.max(np.abs(as_numpy(motion.delta_position_m)))
                ),
                "annual_to_source_angular_frequency_ratio": float(
                    (2.0 * np.pi / (365.256363004 * DAY_S))
                    / source.angular_frequency
                ),
            },
            "snr": {
                key: value
                for key, value in snr.items()
                if key not in {"frequency_hz", "spectra", "psd"}
            },
            "link_metadata": links.metadata,
            "tdi_metadata": tdi_result.metadata,
        },
        "validation": {"all_tdi_finite": True},
        "timings_s": {
            "orbit_setup": orbit_setup_s,
            "worldline_setup": worldline_setup_s,
            "response_and_tdi": response_s,
            "total": float(time.perf_counter() - started),
        },
        "outputs": {
            "figure": str(figure_path),
            "published_figure": (
                None if published_figure is None else str(published_figure)
            ),
            "npz": str(npz_path) if args.save_npz else None,
        },
    }
    summary_path = output_dir / "summary.json"
    write_summary(summary_path, summary)
    print(
        json.dumps(
            {
                "snr_A": snr["snr_A"],
                "snr_E": snr["snr_E"],
                "snr_AE": snr["snr_AE"],
                "response_backend": links.metadata["backend"],
                "tdi_backend": tdi_result.metadata["backend"],
                "summary": str(summary_path),
                "figure": str(figure_path),
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
