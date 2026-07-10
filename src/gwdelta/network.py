"""Network-level response helpers for SSB-frame waveforms."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fastlisa import FastLISAResponseTDI
from .fd_response import C_SI
from .orbits import OrbitSpec, make_orbits_from_spec, orbit_spec_from_dict


DETECTOR_ORDER = ("lisa", "taiji", "tianqin", "bbo")
DETECTOR_ALIASES = {
    "lisa": "lisa",
    "taiji": "taiji",
    "tj": "taiji",
    "tianqin": "tianqin",
    "tq": "tianqin",
    "bbo": "bbo",
}
DEFAULT_DETECTOR_BASES = {
    "lisa": "lisa-simple",
    "taiji": "taiji-simple",
    "tianqin": "tianqin-toy",
    "bbo": "bbo-stage1-toy",
}


@dataclass
class DetectorResponse:
    """Time-domain response for one detector in a network."""

    detector: str
    t: np.ndarray
    channels: dict[str, np.ndarray]
    metadata: dict[str, Any]
    orbit_spec: OrbitSpec
    orbit_setup_s: float
    response_s: float

    @property
    def total_s(self) -> float:
        return float(self.orbit_setup_s + self.response_s)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"t": self.t, **self.channels}


@dataclass
class NetworkResponse:
    """Responses for a detector network."""

    detectors: dict[str, DetectorResponse]
    metadata: dict[str, Any]
    failures: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, detector: str) -> DetectorResponse:
        return self.detectors[normalize_detector_name(detector)]


def normalize_detector_name(detector: str) -> str:
    key = str(detector).strip().lower()
    try:
        return DETECTOR_ALIASES[key]
    except KeyError as exc:
        names = ", ".join(DETECTOR_ORDER)
        raise ValueError(f"unknown detector {detector!r}; available: {names}, or all") from exc


def parse_detector_names(detectors: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize a detector selection."""

    if isinstance(detectors, str):
        items = [item.strip().lower() for item in detectors.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in detectors if str(item).strip()]
    if not items:
        raise ValueError("detectors must not be empty")
    if len(items) == 1 and items[0] == "all":
        return list(DETECTOR_ORDER)

    out: list[str] = []
    for item in items:
        key = normalize_detector_name(item)
        if key not in out:
            out.append(key)
    return out


def load_orbit_overrides(path: str | Path | None) -> dict[str, dict[str, object]]:
    """Load per-detector ``OrbitSpec`` overrides from JSON."""

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if "detectors" in data:
        data = data["detectors"]
    if not isinstance(data, dict):
        raise ValueError("orbit config JSON must be an object keyed by detector name")

    out: dict[str, dict[str, object]] = {}
    for raw_key, value in data.items():
        key = normalize_detector_name(str(raw_key))
        if not isinstance(value, dict):
            raise ValueError(f"orbit config for {raw_key!r} must be an object")
        out[key] = dict(value)
    return out


def orbit_spec_for_detector(
    detector: str,
    *,
    duration: float,
    orbit_dt: float = 600.0,
    orbit_overrides: dict[str, dict[str, object]] | None = None,
) -> OrbitSpec:
    """Build the default simple/toy ``OrbitSpec`` for a detector."""

    name = normalize_detector_name(detector)
    raw = dict((orbit_overrides or {}).get(name, {}))
    raw["base"] = DEFAULT_DETECTOR_BASES[name]
    raw["duration"] = float(duration)
    raw.setdefault("orbit_dt", float(orbit_dt))
    return orbit_spec_from_dict(raw)


def tdi_name_for_response(tdi_generation: str) -> str:
    key = str(tdi_generation).strip().lower()
    if key in {"first", "1", "1st", "1st generation"}:
        return "1st generation"
    if key in {"second", "2", "2nd", "2nd generation"}:
        return "2nd generation"
    raise ValueError("tdi_generation must be first or second")


def to_numpy(value) -> np.ndarray:
    """Convert NumPy/CuPy-like arrays to NumPy."""

    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return np.asarray(value)


def estimate_min_t_buffer(orbits, *, dt: float, order: int) -> float:
    """Match FastLISAResponse's initial-buffer requirement before calling it."""

    x_base = to_numpy(getattr(orbits, "x_base"))
    max_distance_s = float(np.sqrt(np.sum(x_base * x_base, axis=-1)).max() / C_SI)
    projection_buffer = int(max_distance_s) + 4 * int(order)
    check_tdi_buffer = int(100.0 / float(dt)) + 4 * int(order)
    return float((projection_buffer + 2 * check_tdi_buffer + 1) * float(dt))


def resolve_t_buffer(requested: float | None, orbits, *, dt: float, order: int) -> float:
    minimum = estimate_min_t_buffer(orbits, dt=dt, order=order)
    if requested is None:
        return minimum
    if requested <= 0.0:
        raise ValueError("t_buffer must be positive when supplied")
    return max(float(requested), minimum)


def _channel_names(tdi_chan: str) -> tuple[str, ...]:
    key = str(tdi_chan).upper()
    if key not in {"XYZ", "AET", "AE"}:
        raise ValueError("tdi_chan must be XYZ, AET, or AE")
    return tuple(key)


class DetectorNetwork:
    """Compute responses for a named detector network."""

    def __init__(
        self,
        detectors: str | list[str] | tuple[str, ...] = "taiji",
        *,
        orbit_dt: float = 600.0,
        orbit_overrides: dict[str, dict[str, object]] | None = None,
        force_backend: str | None = None,
    ) -> None:
        self.detectors = tuple(parse_detector_names(detectors))
        self.orbit_dt = float(orbit_dt)
        self.orbit_overrides = {} if orbit_overrides is None else dict(orbit_overrides)
        self.force_backend = force_backend

    def orbit_spec(self, detector: str, *, duration: float) -> OrbitSpec:
        return orbit_spec_for_detector(
            detector,
            duration=duration,
            orbit_dt=self.orbit_dt,
            orbit_overrides=self.orbit_overrides,
        )

    def compute_response(
        self,
        t,
        h_plus,
        h_cross,
        *,
        lam: float,
        beta: float,
        tdi_generation: str = "second",
        tdi_chan: str = "AE",
        order: int = 15,
        t_buffer: float | None = None,
        trim_garbage: bool = True,
        orbit_margin_s: float = 1200.0,
        force_backend: str | None = None,
        skip_unavailable: bool = False,
    ) -> NetworkResponse:
        """Compute detector responses to one SSB-frame polarization time series."""

        t_np = np.asarray(t, dtype=float)
        if t_np.ndim != 1 or len(t_np) < 2:
            raise ValueError("t must be a one-dimensional array with at least two samples")
        dt = float(np.median(np.diff(t_np)))
        if not np.allclose(np.diff(t_np), dt, rtol=1.0e-10, atol=max(1.0e-12, abs(dt) * 1.0e-12)):
            raise ValueError("t must be uniformly sampled")

        duration = float(t_np[-1] - t_np[0] + orbit_margin_s)
        backend = self.force_backend if force_backend is None else force_backend
        tdi = tdi_name_for_response(tdi_generation)
        names = _channel_names(tdi_chan)

        responses: dict[str, DetectorResponse] = {}
        failures: dict[str, str] = {}
        for detector in self.detectors:
            tic = time.perf_counter()
            try:
                spec = self.orbit_spec(detector, duration=duration)
                orbits = make_orbits_from_spec(spec, force_backend=backend)
                orbit_setup_s = time.perf_counter() - tic

                buffer_s = resolve_t_buffer(t_buffer, orbits, dt=dt, order=order)
                if trim_garbage and 2 * int(buffer_s / dt) >= len(t_np):
                    raise ValueError(
                        "trim_garbage would remove all samples; increase duration, decrease dt, "
                        f"or disable trimming. Required t_buffer is {buffer_s:g} s."
                    )

                response = FastLISAResponseTDI(
                    orbits=orbits,
                    order=order,
                    tdi=tdi,
                    tdi_chan=tdi_chan,
                    force_backend=backend,
                    t_buffer=buffer_s,
                    trim_garbage=trim_garbage,
                )
                tic_response = time.perf_counter()
                result = response.compute(t_np, h_plus, h_cross, lam=lam, beta=beta)
                response_s = time.perf_counter() - tic_response
                data = result.as_numpy()
                channels = {name: np.asarray(data[name], dtype=float) for name in names}
                metadata = dict(result.metadata)
                metadata.update(
                    {
                        "detector": detector,
                        "requested_t_buffer": t_buffer,
                        "auto_min_t_buffer": estimate_min_t_buffer(orbits, dt=dt, order=order),
                        "orbit_base": spec.base,
                    }
                )
                responses[detector] = DetectorResponse(
                    detector=detector,
                    t=np.asarray(data["t"], dtype=float),
                    channels=channels,
                    metadata=metadata,
                    orbit_spec=spec,
                    orbit_setup_s=float(orbit_setup_s),
                    response_s=float(response_s),
                )
            except Exception as exc:
                failures[detector] = f"{type(exc).__name__}: {exc}"
                if not skip_unavailable:
                    raise

        metadata = {
            "detectors": list(self.detectors),
            "tdi_generation": str(tdi_generation),
            "tdi": tdi,
            "tdi_chan": str(tdi_chan).upper(),
            "order": int(order),
            "dt_s": dt,
            "input_samples": int(len(t_np)),
            "orbit_dt_s": self.orbit_dt,
            "orbit_margin_s": float(orbit_margin_s),
            "force_backend": backend,
        }
        return NetworkResponse(detectors=responses, failures=failures, metadata=metadata)


__all__ = [
    "DEFAULT_DETECTOR_BASES",
    "DETECTOR_ALIASES",
    "DETECTOR_ORDER",
    "DetectorNetwork",
    "DetectorResponse",
    "NetworkResponse",
    "estimate_min_t_buffer",
    "load_orbit_overrides",
    "normalize_detector_name",
    "orbit_spec_for_detector",
    "parse_detector_names",
    "resolve_t_buffer",
    "tdi_name_for_response",
    "to_numpy",
]
