"""One-year ESA-LISA response to a uniformly moving point mass."""

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
    as_numpy,
    build_time_grids,
    channel_spectra,
    configure_cuda_if_needed,
    configure_latex_roman,
    display_scale,
    make_esa_lisa_orbits,
    make_response_engines,
    normalized,
    tdi_channels_numpy,
    write_summary,
)
from gwdelta import C_SI, G_SI, UniformMovingPointMass, select_array_backend


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lisa_uniform_moving_point_mass"
README_FIGURE_NAME = "lisa_uniform_moving_point_mass_demo.png"


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
    parser.add_argument("--point-mass-kg", type=float, default=1.0e20)
    parser.add_argument("--flyby-relative-speed-m-s", type=float, default=3.0e5)
    parser.add_argument("--flyby-impact-parameter-m", type=float, default=5.0e9)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument(
        "--publish-figure",
        action="store_true",
        help="also write the reproducible README figure under docs/figures",
    )
    return parser.parse_args()


def interpolate_orbit_worldlines(
    orbits, query_time_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t_base = as_numpy(orbits.t_base).astype(float, copy=False)
    x_base = as_numpy(orbits.x_base).astype(float, copy=False)
    v_base = as_numpy(orbits.v_base).astype(float, copy=False)
    if query_time_s[0] < t_base[0] or query_time_s[-1] > t_base[-1]:
        raise ValueError("worldline query lies outside the ESA orbit interval")
    position = CubicSpline(t_base, x_base, axis=0)(query_time_s)
    velocity = CubicSpline(t_base, v_base, axis=0)(query_time_s)
    return position, velocity


def build_moving_source(
    args: argparse.Namespace,
    encounter_time_s: float,
    background_position_m: np.ndarray,
    background_velocity_m_s: np.ndarray,
) -> tuple[UniformMovingPointMass, dict[str, object]]:
    if args.flyby_relative_speed_m_s <= 0.0:
        raise ValueError("flyby_relative_speed_m_s must be positive")
    if args.flyby_impact_parameter_m <= 0.0:
        raise ValueError("flyby_impact_parameter_m must be positive")
    if args.point_mass_kg <= 0.0:
        raise ValueError("point_mass_kg must be positive")

    center_position = np.mean(background_position_m, axis=0)
    center_velocity = np.mean(background_velocity_m_s, axis=0)
    relative_direction = np.asarray([0.0, 0.0, 1.0])
    radial = normalized(center_position, name="LISA center radial direction")
    impact_direction = radial - np.dot(radial, relative_direction) * relative_direction
    impact_direction = normalized(impact_direction, name="flyby impact direction")
    source_position = center_position + args.flyby_impact_parameter_m * impact_direction
    source_velocity = (
        center_velocity + args.flyby_relative_speed_m_s * relative_direction
    )
    source = UniformMovingPointMass(
        rest_mass_kg=args.point_mass_kg,
        position_at_reference_m=tuple(source_position),
        velocity_m_s=tuple(source_velocity),
        reference_time_s=encounter_time_s,
    )
    return source, {
        "encounter_time_s": float(encounter_time_s),
        "constellation_center_position_m": center_position.tolist(),
        "constellation_center_velocity_m_s": center_velocity.tolist(),
        "relative_velocity_direction": relative_direction.tolist(),
        "impact_direction": impact_direction.tolist(),
        "position_at_reference_m": list(source.position_at_reference_m),
        "velocity_m_s": list(source.velocity_m_s),
        "source_beta": source.beta,
        "source_lorentz_factor": source.lorentz_factor,
    }


def compute_tdi_components(
    tdi_engine,
    time_s: np.ndarray,
    components: dict[str, object],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    channels = {}
    metadata = {}
    for name, links in components.items():
        result = tdi_engine.compute_links(time_s, links)
        channels[name] = tdi_channels_numpy(result)
        metadata[name] = result.metadata
    return channels, metadata


def minimum_link_rho(
    source: UniformMovingPointMass, geometry, chunk_size: int
) -> float:
    beta = source.beta_vector
    gamma_squared = source.lorentz_factor**2
    minimum = np.inf
    nt = len(geometry.t_reception_s)
    for start in range(0, nt, chunk_size):
        stop = min(start + chunk_size, nt)
        t_e = geometry.t_emission_s[:, start:stop]
        x_e = geometry.x_emission_m[:, start:stop]
        direction = geometry.direction[:, start:stop]
        light_time = geometry.light_time_s[:, start:stop]
        length = geometry.chord_length_m[:, start:stop]
        s_e = x_e - source.position(t_e)
        q = direction - (C_SI * light_time / length)[..., np.newaxis] * beta
        beta_dot_q = np.einsum("...i,i->...", q, beta)
        beta_dot_e = np.einsum("...i,i->...", s_e, beta)
        a = np.einsum("...i,...i->...", q, q) + gamma_squared * beta_dot_q**2
        b = (
            np.einsum("...i,...i->...", s_e, q)
            + gamma_squared * beta_dot_e * beta_dot_q
        )
        ell = np.clip(-b / a, 0.0, length)
        closest = s_e + ell[..., np.newaxis] * q
        beta_dot_closest = np.einsum("...i,i->...", closest, beta)
        rho_squared = (
            np.einsum("...i,...i->...", closest, closest)
            + gamma_squared * beta_dot_closest**2
        )
        minimum = min(minimum, float(np.sqrt(np.min(rho_squared))))
    return minimum


def make_figure(
    path: Path,
    *,
    components: dict[str, dict[str, np.ndarray]],
    encounter_time_s: float,
    flyby_timescale_s: float,
) -> None:
    import matplotlib.pyplot as plt

    configure_latex_roman()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)

    total = components["total"]
    mask = np.abs(total["t"] - encounter_time_s) <= 6.0 * flyby_timescale_s
    scale, exponent = display_scale(total["A"][mask], total["E"][mask])
    colors = {"A": "#D55E00", "E": "#0072B2"}
    for channel in ("A", "E"):
        axes[0].plot(
            (total["t"][mask] - encounter_time_s) / DAY_S,
            total[channel][mask] / scale,
            color=colors[channel],
            lw=1.0,
            label=rf"${channel}$",
        )
    axes[0].set_xlabel(r"$t-t_{\rm ref}\,[{\rm day}]$")
    axes[0].set_ylabel(rf"$U(t)\ [10^{{{exponent}}}]$")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")

    styles = {
        "total": ("#222222", "-"),
        "worldline": ("#009E73", "--"),
        "direct": ("#CC79A7", ":"),
    }
    for name in ("total", "worldline", "direct"):
        frequency_hz, spectra = channel_spectra(components[name]["t"], components[name])
        combined = np.sqrt(np.abs(spectra["A"]) ** 2 + np.abs(spectra["E"]) ** 2)
        positive = frequency_hz > 0.0
        color, line_style = styles[name]
        axes[1].loglog(
            frequency_hz[positive],
            combined[positive],
            color=color,
            ls=line_style,
            lw=1.0,
            label={
                "direct": "photon propagation",
                "worldline": "endpoint velocity",
                "total": "photon propagation + endpoint velocity",
            }[name],
        )
    axes[1].set_xlim(max(frequency_hz[1], 1.0e-8), min(3.0e-3, frequency_hz[-1]))
    axes[1].set_xlabel(r"$f\,[{\rm Hz}]$")
    axes[1].set_ylabel(r"$[|\widetilde A|^2+|\widetilde E|^2]^{1/2}\,[{\rm s}]$")
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
    background_position, background_velocity = interpolate_orbit_worldlines(
        orbits, motion_time
    )
    minimum_emission_time = float(response_time[0] - np.max(as_numpy(orbits.ltt_base)))
    if minimum_emission_time < motion_time[0]:
        raise ValueError("motion grid does not cover the earliest emission event")

    trim_samples = int(args.t_buffer / args.dt)
    output_start = response_time[trim_samples]
    output_stop = response_time[-trim_samples - 1]
    encounter_time_s = 0.5 * (output_start + output_stop)
    encounter_position = CubicSpline(motion_time, background_position, axis=0)(
        encounter_time_s
    )
    encounter_velocity = CubicSpline(motion_time, background_velocity, axis=0)(
        encounter_time_s
    )
    source, source_geometry = build_moving_source(
        args,
        encounter_time_s,
        encounter_position,
        encounter_velocity,
    )

    array_backend = select_array_backend(
        force="cupy" if args.response_backend == "cuda12x" else "cpu"
    )
    tic = time.perf_counter()
    motion = source.integrate_test_mass_motion(
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
    minimum_rho_m = minimum_link_rho(source, links.geometry, max(1024, args.chunk_size))
    components, tdi_metadata = compute_tdi_components(
        tdi_engine,
        response_time,
        {
            "direct": links.direct,
            "worldline": links.worldline,
            "total": links.total,
        },
    )
    response_s = time.perf_counter() - tic
    if any(len(channels["t"]) != analysis_samples for channels in components.values()):
        raise RuntimeError(
            "TDI trimming did not produce the requested observation length"
        )
    reference_time = components["total"]["t"]
    if any(
        not np.array_equal(channels["t"], reference_time)
        for channels in components.values()
    ):
        raise RuntimeError("TDI component time grids differ")
    if not all(
        np.all(np.isfinite(channels[channel]))
        for channels in components.values()
        for channel in ("A", "E")
    ):
        raise RuntimeError("the point-mass TDI response contains non-finite values")

    link_sum_error = float(
        as_numpy(
            array_backend.xp.linalg.norm(links.total - links.direct - links.worldline)
        )
    )
    total_norm = np.sqrt(
        np.linalg.norm(components["total"]["A"]) ** 2
        + np.linalg.norm(components["total"]["E"]) ** 2
    )
    component_error = np.sqrt(
        np.linalg.norm(
            components["total"]["A"]
            - components["direct"]["A"]
            - components["worldline"]["A"]
        )
        ** 2
        + np.linalg.norm(
            components["total"]["E"]
            - components["direct"]["E"]
            - components["worldline"]["E"]
        )
        ** 2
    )
    tdi_linearity_relative_l2 = float(
        component_error / max(total_norm, np.finfo(float).tiny)
    )

    flyby_timescale_s = args.flyby_impact_parameter_m / args.flyby_relative_speed_m_s
    figure_path = output_dir / README_FIGURE_NAME
    make_figure(
        figure_path,
        components=components,
        encounter_time_s=encounter_time_s,
        flyby_timescale_s=flyby_timescale_s,
    )
    published_figure = None
    if args.publish_figure:
        published_figure = REPO_ROOT / "docs" / "figures" / README_FIGURE_NAME
        make_figure(
            published_figure,
            components=components,
            encounter_time_s=encounter_time_s,
            flyby_timescale_s=flyby_timescale_s,
        )

    npz_path = output_dir / "lisa_uniform_moving_point_mass_demo.npz"
    if args.save_npz:
        np.savez_compressed(
            npz_path,
            t_seconds=reference_time,
            direct_A=components["direct"]["A"],
            direct_E=components["direct"]["E"],
            worldline_A=components["worldline"]["A"],
            worldline_E=components["worldline"]["E"],
            total_A=components["total"]["A"],
            total_E=components["total"]["E"],
        )

    summary = {
        "example": "ESA LISA uniform moving point mass",
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
            "samples": int(len(reference_time)),
            "dt_s": float(args.dt),
            "duration_s": float(len(reference_time) * args.dt),
            "start_time_s": float(reference_time[0]),
            "stop_time_s": float(reference_time[-1]),
            "tdi_generation": "second",
            "tdi_channels": "AE",
        },
        "uniform_moving_point_mass": {
            "rest_mass_kg": float(args.point_mass_kg),
            "relative_speed_m_s": float(args.flyby_relative_speed_m_s),
            "impact_parameter_m": float(args.flyby_impact_parameter_m),
            "flyby_timescale_s": float(flyby_timescale_s),
            "minimum_link_rho_m": float(minimum_rho_m),
            "maximum_GM_over_c2rho": float(
                G_SI * args.point_mass_kg / (C_SI**2 * minimum_rho_m)
            ),
            "geometry": source_geometry,
            "response_scope": "analytic photon-propagation plus leading endpoint-velocity worldline response",
            "link_metadata": links.metadata,
            "tdi_metadata": tdi_metadata,
        },
        "validation": {
            "link_total_minus_components_l2": link_sum_error,
            "tdi_total_minus_components_relative_l2": tdi_linearity_relative_l2,
            "all_tdi_finite": True,
        },
        "timings_s": {
            "orbit_setup": orbit_setup_s,
            "worldline": worldline_setup_s,
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
                "minimum_link_rho_m": minimum_rho_m,
                "maximum_GM_over_c2rho": summary["uniform_moving_point_mass"][
                    "maximum_GM_over_c2rho"
                ],
                "tdi_linearity_relative_l2": tdi_linearity_relative_l2,
                "response_backend": links.metadata["backend"],
                "tdi_backend": tdi_metadata["total"]["backend"],
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
