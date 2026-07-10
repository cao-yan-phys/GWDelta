"""Custom orbit adapters compatible with FastLISAResponse/lisatools."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import interpolate

from .cuda_runtime import backend_wants_cuda, ensure_cuda_dll_directories

C_SI = 299792458.0
AU_SI = 1.496e11
DAY_SI = 86400.0
SIDEREAL_YEAR_S = 31558149.763545603
EARTH_MU_SI = 3.986004418e14
LISA_NOMINAL_ARM_M = 2.5e9
TAIJI_NOMINAL_ARM_M = 3.0e9
TIANQIN_GEOCENTRIC_RADIUS_M = 1.0e8
TIANQIN_NOMINAL_ARM_M = np.sqrt(3.0) * TIANQIN_GEOCENTRIC_RADIUS_M
TIANQIN_J0806_LON = np.deg2rad(120.5)
TIANQIN_J0806_LAT = np.deg2rad(-4.7)
BBO_STAGE1_ARM_M = 5.0e7
BBO_STAGE1_CENTER_PHASE = np.deg2rad(-20.0)
BBO_STAGE1_PLANE_INCLINATION = np.deg2rad(60.0)
BBO_STAGE1_CARTWHEEL_PHASE = np.deg2rad(90.0)
LISA_TRAILING_CENTER_PHASE = np.deg2rad(-20.0)
TAIJI_AHEAD_CENTER_PHASE = np.deg2rad(20.0)
LISA_SIMPLE_CARTWHEEL_PHASE = np.deg2rad(90.0)
TAIJI_SIMPLE_CARTWHEEL_PHASE = np.deg2rad(-90.0)
DEFAULT_LINKS = [12, 23, 31, 13, 32, 21]
DEFAULT_SC = [1, 2, 3]
LINEAR_INTERP_TIMESTEP = 600.0


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value).expanduser()


def _default_taiji_triangle_orbit_dir() -> Path | None:
    path = _env_path("GWDELTA_TAIJI_TRIANGLE_ORBIT_DIR")
    if path is not None:
        return path
    root = _env_path("GWDELTA_ORBIT_DATA_DIR")
    return None if root is None else root / "TaijiEqualArmOrbit"


def _default_taiji_accurate_orbit_dir() -> Path | None:
    path = _env_path("GWDELTA_TAIJI_ACCURATE_ORBIT_DIR")
    if path is not None:
        return path
    root = _env_path("GWDELTA_ORBIT_DATA_DIR")
    return None if root is None else root / "MicroSateOrbitEclipticTCB"

try:  # pragma: no cover - optional runtime dependency
    from lisatools.detector import Orbits
    from lisatools.utils.parallelbase import LISAToolsParallelModule

    _HAS_LISATOOLS = True
except Exception:  # pragma: no cover - optional runtime dependency
    _HAS_LISATOOLS = False

    class Orbits:  # type: ignore[no-redef]
        pass

    class LISAToolsParallelModule:  # type: ignore[no-redef]
        def __init__(self, force_backend=None):
            self.backend = None


@dataclass(frozen=True)
class OrbitArrays:
    """Host arrays defining a sampled triangular detector orbit."""

    t: np.ndarray
    x: np.ndarray
    n: np.ndarray
    ltt: np.ndarray
    v: np.ndarray
    armlength: float
    links: tuple[int, ...]


@dataclass(frozen=True)
class OrbitSpec:
    """User-facing specification for a FastLISAResponse orbit.

    ``base`` can be one of ``esa``, ``equal-armlength``, ``lisa-simple``,
    ``taiji-simple``, ``taiji-triangle``, ``taiji-accurate``, ``tianqin-toy``,
    ``bbo-stage1-toy``, or ``file``.
    File orbits use ``path`` and currently support ``npz`` or headered ``csv``
    inputs.
    """

    base: str = "esa"
    path: str | None = None
    orbit_dir: str | None = None
    duration: float | None = None
    orbit_dt: float = LINEAR_INTERP_TIMESTEP
    time_offset: float = 0.0
    rotate_z: float = 0.0
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    armlength: float | None = None
    links: tuple[int, ...] = tuple(DEFAULT_LINKS)
    force_sampled: bool = False
    max_rows: int | None = None
    file_format: str | None = None
    time_unit: str = "s"
    length_unit: str = "m"
    preserve_ltt: bool = False
    center_radius: float | None = None
    center_period: float | None = None
    center_phase: float | None = None
    cartwheel_period: float | None = None
    cartwheel_phase: float | None = None
    plane_inclination: float | None = None
    normal_lon: float | None = None
    normal_lat: float | None = None
    geocentric_radius: float | None = None
    use_project_phase_defaults: bool = True


def _unit_factor(unit: str, table: dict[str, float], kind: str) -> float:
    key = unit.strip().lower()
    if key not in table:
        raise ValueError(f"unsupported {kind} unit {unit!r}; choices are {sorted(table)}")
    return table[key]


def _time_unit_factor(unit: str) -> float:
    return _unit_factor(
        unit,
        {
            "s": 1.0,
            "sec": 1.0,
            "second": 1.0,
            "seconds": 1.0,
            "day": DAY_SI,
            "days": DAY_SI,
            "yr": 31558149.763545603,
            "year": 31558149.763545603,
            "sidereal_year": 31558149.763545603,
        },
        "time",
    )


def _length_unit_factor(unit: str) -> float:
    return _unit_factor(
        unit,
        {
            "m": 1.0,
            "meter": 1.0,
            "meters": 1.0,
            "km": 1.0e3,
            "au": AU_SI,
        },
        "length",
    )


def _validate_links(links: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    links_tuple = tuple(int(link) for link in links)
    if links_tuple != tuple(DEFAULT_LINKS):
        raise ValueError(
            "FastLISAResponse's current C++ orbit backend assumes link order "
            "12,23,31,13,32,21; reorder input files to this order before loading"
        )
    return links_tuple


def _link_indices(links: list[int] | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    links_tuple = _validate_links(links)
    receivers = np.asarray([int(str(link)[0]) - 1 for link in links_tuple], dtype=np.int32)
    emitters = np.asarray([int(str(link)[1]) - 1 for link in links_tuple], dtype=np.int32)
    return receivers, emitters


def build_orbit_arrays(
    t,
    x,
    *,
    links: list[int] | tuple[int, ...] = tuple(DEFAULT_LINKS),
    armlength: float | None = None,
    v=None,
    ltt=None,
) -> OrbitArrays:
    """Build link vectors, light-travel times, and velocities from spacecraft positions.

    ``x`` must have shape ``(nt, 3, 3)``: time, spacecraft, Cartesian component.
    Link ``ij`` is interpreted in the lisatools convention as receiver ``i`` and
    emitter ``j``.  The link unit vector points from emitter to receiver.
    """

    t_arr = np.asarray(t, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if t_arr.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if x_arr.shape != (len(t_arr), 3, 3):
        raise ValueError("x must have shape (len(t), 3, 3)")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("t must be strictly increasing")

    links_tuple = _validate_links(links)
    receivers, emitters = _link_indices(links_tuple)
    delta = x_arr[:, receivers, :] - x_arr[:, emitters, :]
    lengths = np.linalg.norm(delta, axis=-1)
    if np.any(lengths <= 0.0):
        raise ValueError("spacecraft positions produced a zero-length arm")
    n_arr = delta / lengths[..., None]
    if ltt is None:
        ltt_arr = lengths / C_SI
    else:
        ltt_arr = np.asarray(ltt, dtype=float)
        if ltt_arr.shape != lengths.shape:
            raise ValueError("ltt must have shape (len(t), len(links))")
    if v is None:
        v_arr = np.gradient(x_arr, t_arr, axis=0, edge_order=1)
    else:
        v_arr = np.asarray(v, dtype=float)
        if v_arr.shape != x_arr.shape:
            raise ValueError("v must have the same shape as x")
    arm = float(np.median(lengths)) if armlength is None else float(armlength)
    return OrbitArrays(t=t_arr, x=x_arr, n=n_arr, ltt=ltt_arr, v=v_arr, armlength=arm, links=links_tuple)


def orbit_spec_from_dict(data: dict[str, Any]) -> OrbitSpec:
    """Create an ``OrbitSpec`` from a plain dictionary.

    Accepted aliases are intentionally human-facing: ``type`` for ``base``,
    ``file`` for ``path``, ``rotate_z_deg`` for a degree-valued rotation, and
    ``translation_m`` for ``translate``.
    """

    raw = dict(data)
    if "type" in raw and "base" not in raw:
        raw["base"] = raw.pop("type")
    if "file" in raw and "path" not in raw:
        raw["path"] = raw.pop("file")
    if "translation_m" in raw and "translate" not in raw:
        raw["translate"] = raw.pop("translation_m")
    if "armlength_m" in raw and "armlength" not in raw:
        raw["armlength"] = raw.pop("armlength_m")
    if "rotate_z_deg" in raw:
        if "rotate_z" in raw:
            raise ValueError("use only one of rotate_z and rotate_z_deg")
        raw["rotate_z"] = np.deg2rad(float(raw.pop("rotate_z_deg")))
    for name in ("center_phase", "cartwheel_phase", "plane_inclination", "normal_lon", "normal_lat"):
        deg_name = name + "_deg"
        if deg_name in raw:
            if name in raw:
                raise ValueError(f"use only one of {name} and {deg_name}")
            raw[name] = np.deg2rad(float(raw.pop(deg_name)))
    if "links" in raw:
        raw["links"] = tuple(int(item) for item in raw["links"])
    if "translate" in raw:
        translate = tuple(float(item) for item in raw["translate"])
        if len(translate) != 3:
            raise ValueError("translate must contain three numbers")
        raw["translate"] = translate
    return OrbitSpec(**raw)


def load_orbit_spec(path: str | Path) -> OrbitSpec:
    """Load an orbit specification from a JSON file."""

    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    spec = orbit_spec_from_dict(data)
    if spec.path is not None and not Path(spec.path).is_absolute():
        spec = OrbitSpec(**{**spec.__dict__, "path": str((spec_path.parent / spec.path).resolve())})
    if spec.orbit_dir is not None and not Path(spec.orbit_dir).is_absolute():
        spec = OrbitSpec(**{**spec.__dict__, "orbit_dir": str((spec_path.parent / spec.orbit_dir).resolve())})
    return spec


def _has_sampling_or_transform(spec: OrbitSpec) -> bool:
    return (
        spec.force_sampled
        or spec.duration is not None
        or spec.time_offset != 0.0
        or spec.rotate_z != 0.0
        or spec.translate != (0.0, 0.0, 0.0)
        or spec.scale != 1.0
        or spec.armlength is not None
    )


def _apply_position_transforms(
    x: np.ndarray,
    *,
    rotate_z: float = 0.0,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    out = np.asarray(x, dtype=float) * float(scale)
    out = _rotate_z(out, float(rotate_z))
    shift = np.asarray(translate, dtype=float)
    if shift.shape != (3,):
        raise ValueError("translate must be a 3-vector")
    return out + shift


def _apply_velocity_transforms(v: np.ndarray | None, *, rotate_z: float = 0.0, scale: float = 1.0) -> np.ndarray | None:
    if v is None:
        return None
    out = np.asarray(v, dtype=float) * float(scale)
    return _rotate_z(out, float(rotate_z))


def _apply_ltt_scale(ltt: np.ndarray | None, *, scale: float = 1.0, preserve_ltt: bool = False) -> np.ndarray | None:
    if ltt is None:
        return None
    if preserve_ltt:
        return np.asarray(ltt, dtype=float)
    return np.asarray(ltt, dtype=float) * abs(float(scale))


def _center_phase_from_positions(x: np.ndarray) -> float:
    center = np.mean(np.asarray(x, dtype=float)[0], axis=0)
    return float(np.arctan2(center[1], center[0]))


def _align_arrays_center_phase(arrays: OrbitArrays, target_phase: float | None) -> OrbitArrays:
    if target_phase is None:
        return arrays
    current = _center_phase_from_positions(arrays.x)
    delta = float(target_phase) - current
    x = _rotate_z(arrays.x, delta)
    v = _rotate_z(arrays.v, delta)
    return build_orbit_arrays(arrays.t, x, links=arrays.links, armlength=arrays.armlength, v=v, ltt=arrays.ltt)


def _default_center_phase_for_base(base: str, spec: OrbitSpec) -> float | None:
    if spec.center_phase is not None:
        return spec.center_phase
    if not spec.use_project_phase_defaults:
        return None
    if base in ("esa", "equal-armlength", "equal_armlength", "equal", "lisa-simple"):
        return LISA_TRAILING_CENTER_PHASE
    if base in ("taiji-simple", "taiji-triangle", "taiji-accurate"):
        return TAIJI_AHEAD_CENTER_PHASE
    return None


def _sample_orbit_arrays_from_lisatools(
    base_orbits: Orbits,
    t_query: np.ndarray,
    *,
    t_local: np.ndarray,
    rotate_z: float = 0.0,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    armlength: float | None = None,
    preserve_ltt: bool = False,
) -> OrbitArrays:
    t_base = np.asarray(base_orbits.t_base, dtype=float)
    if np.any(t_query < t_base[0]) or np.any(t_query > t_base[-1]):
        raise ValueError("requested sampled orbit lies outside the base orbit")
    x = _interpolate_vector_series(t_base, np.asarray(base_orbits.x_base, dtype=float), t_query)
    v = _interpolate_vector_series(t_base, np.asarray(base_orbits.v_base, dtype=float), t_query)
    try:
        ltt_data = np.asarray(base_orbits.ltt_base, dtype=float)
    except Exception:
        ltt_data = None
    ltt = None if ltt_data is None else _interpolate_array_series(t_base, ltt_data, t_query)
    x = _apply_position_transforms(x, rotate_z=rotate_z, translate=translate, scale=scale)
    v = _apply_velocity_transforms(v, rotate_z=rotate_z, scale=scale)
    ltt = _apply_ltt_scale(ltt, scale=scale, preserve_ltt=preserve_ltt)
    arm = armlength
    if arm is None:
        arm = float(base_orbits.armlength) * abs(float(scale))
    return build_orbit_arrays(t_local, x, links=tuple(base_orbits.LINKS), armlength=arm, v=v, ltt=ltt)


class SampledOrbits(Orbits):
    """An in-memory ``lisatools.detector.Orbits`` subclass for custom orbits."""

    def __init__(
        self,
        t,
        x,
        *,
        links: list[int] | tuple[int, ...] = tuple(DEFAULT_LINKS),
        armlength: float | None = None,
        v=None,
        ltt=None,
        force_backend: str | None = None,
    ) -> None:
        if not _HAS_LISATOOLS:
            raise ImportError(
                "SampledOrbits requires lisaanalysistools. Install it or an associated cuda package first."
            )
        if backend_wants_cuda(force_backend):
            ensure_cuda_dll_directories()
        arrays = build_orbit_arrays(t, x, links=links, armlength=armlength, v=v, ltt=ltt)
        self._arrays = arrays
        self._armlength = arrays.armlength
        self.configured = False
        self._dt = None
        self._pycppdetector_args = None
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @property
    def xp(self):
        return self.backend.xp

    @property
    def armlength(self) -> float:
        return self._armlength

    @armlength.setter
    def armlength(self, armlength: float) -> None:
        self._armlength = float(armlength)

    @property
    def LINKS(self) -> list[int]:
        return list(self._arrays.links)

    @property
    def SC(self) -> list[int]:
        return list(DEFAULT_SC)

    @property
    def link_space_craft_r(self) -> list[int]:
        return [int(str(link_i)[0]) for link_i in self.LINKS]

    @property
    def link_space_craft_e(self) -> list[int]:
        return [int(str(link_i)[1]) for link_i in self.LINKS]

    @property
    def size_base(self) -> int:
        return len(self._arrays.t)

    @property
    def dt_base(self) -> float:
        return float(np.median(np.diff(self._arrays.t)))

    @property
    def t_base(self) -> np.ndarray:
        return self._arrays.t

    @property
    def x_base(self) -> np.ndarray:
        return self._arrays.x

    @property
    def n_base(self) -> np.ndarray:
        return self._arrays.n

    @property
    def ltt_base(self) -> np.ndarray:
        return self._arrays.ltt

    @property
    def v_base(self) -> np.ndarray:
        return self._arrays.v

    @property
    def t(self) -> np.ndarray:
        self._check_configured()
        return self._t

    @t.setter
    def t(self, t: np.ndarray) -> None:
        self._t = np.asarray(t, dtype=float)

    @property
    def x(self) -> np.ndarray:
        self._check_configured()
        return self._x

    @property
    def n(self) -> np.ndarray:
        self._check_configured()
        return self._n

    @property
    def ltt(self) -> np.ndarray:
        self._check_configured()
        return self._ltt

    @property
    def v(self) -> np.ndarray:
        self._check_configured()
        return self._v

    @property
    def dt(self) -> float:
        if self._dt is None:
            raise ValueError("dt is not available until configure() creates an equally spaced grid")
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self._dt = float(dt)

    @property
    def pycppdetector_args(self) -> list[Any] | None:
        return self._pycppdetector_args

    @pycppdetector_args.setter
    def pycppdetector_args(self, args) -> None:
        self._pycppdetector_args = args

    def _check_configured(self) -> None:
        if not self.configured:
            raise ValueError("Cannot request configured orbit arrays before configure()")

    def configure(self, t_arr=None, dt: float | None = None, linear_interp_setup: bool = False) -> None:
        """Configure the custom orbit on the grid expected by FastLISAResponse."""

        if linear_interp_setup:
            dt = LINEAR_INTERP_TIMESTEP
            nobs = int(self.t_base[-1] / dt)
            t_arr = np.arange(nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])
        elif t_arr is not None:
            t_arr = np.asarray(t_arr, dtype=float)
            if len(t_arr) < 2:
                raise ValueError("t_arr must contain at least two samples")
            dt = float(abs(t_arr[1] - t_arr[0]))
        elif dt is not None:
            nobs = int(self.t_base[-1] / dt)
            t_arr = np.arange(nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])
        else:
            t_arr = self.t_base.copy()
            dt = self.dt_base

        if t_arr[0] < self.t_base[0] or t_arr[-1] > self.t_base[-1]:
            raise ValueError("requested configured orbit grid lies outside the sampled base orbit")

        self.t = t_arr.copy()
        x_orig = self.t_base
        for which, arr in (("ltt", self.ltt_base), ("x", self.x_base), ("n", self.n_base), ("v", self.v_base)):
            flat = arr.reshape(len(x_orig), -1)
            out_flat = np.zeros((len(t_arr), flat.shape[-1]))
            for i in range(flat.shape[-1]):
                out_flat[:, i] = interpolate.CubicSpline(x_orig, flat[:, i])(t_arr)
            setattr(self, "_" + which, out_flat.reshape((len(t_arr),) + arr.shape[1:]))

        receivers = np.asarray(self.link_space_craft_r, dtype=np.int32)
        emitters = np.asarray(self.link_space_craft_e, dtype=np.int32)
        links = np.asarray(self.LINKS, dtype=np.int32)
        self.configured = True
        self.dt = float(dt)
        self.pycppdetector_args = [
            self.dt,
            len(self.t),
            self.xp.asarray(self.n.flatten().copy()),
            self.xp.asarray(self.ltt.flatten().copy()),
            self.xp.asarray(self.x.flatten().copy()),
            self.xp.asarray(links),
            self.xp.asarray(receivers),
            self.xp.asarray(emitters),
            self.armlength,
        ]


def _sample_base_positions(base_orbits: Orbits, t_query: np.ndarray) -> np.ndarray:
    """Interpolate spacecraft positions from a lisatools orbit object."""

    t_base = np.asarray(base_orbits.t_base, dtype=float)
    x_base = np.asarray(base_orbits.x_base, dtype=float)
    if np.any(t_query < t_base[0]) or np.any(t_query > t_base[-1]):
        raise ValueError(
            "requested time-shifted orbit lies outside the base orbit; "
            "reduce duration/time_offset or use a longer orbit file"
        )
    flat = x_base.reshape(len(t_base), -1)
    out = np.empty((len(t_query), flat.shape[-1]), dtype=float)
    for i in range(flat.shape[-1]):
        out[:, i] = interpolate.CubicSpline(t_base, flat[:, i])(t_query)
    return out.reshape((len(t_query),) + x_base.shape[1:])


def _rotate_z(vectors: np.ndarray, angle: float) -> np.ndarray:
    """Rigidly rotate Cartesian vectors about the SSB z axis."""

    if angle == 0.0:
        return vectors
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.einsum("ij,...j->...i", rot, vectors)


def _unit_from_ecliptic(lon: float, lat: float) -> np.ndarray:
    cos_lat = float(np.cos(lat))
    return np.asarray(
        [
            cos_lat * float(np.cos(lon)),
            cos_lat * float(np.sin(lon)),
            float(np.sin(lat)),
        ],
        dtype=float,
    )


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 0.0:
        raise ValueError("plane normal must be nonzero")
    n = n / norm
    z_axis = np.asarray([0.0, 0.0, 1.0])
    u = np.cross(z_axis, n)
    if np.linalg.norm(u) < 1.0e-14:
        u = np.asarray([1.0, 0.0, 0.0])
    else:
        u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u, v, n


def _circular_heliocentric_center(
    t: np.ndarray,
    *,
    radius: float = AU_SI,
    period: float = SIDEREAL_YEAR_S,
    phase: float = 0.0,
) -> np.ndarray:
    alpha = 2.0 * np.pi * np.asarray(t, dtype=float) / float(period) + float(phase)
    return float(radius) * np.column_stack([np.cos(alpha), np.sin(alpha), np.zeros_like(alpha)])


def _fixed_normal_rigid_triangle(
    t: np.ndarray,
    center: np.ndarray,
    *,
    arm_length: float,
    normal: np.ndarray,
    period: float,
    phase: float = 0.0,
) -> np.ndarray:
    u, v, _n = _basis_from_normal(normal)
    rho = float(arm_length) / np.sqrt(3.0)
    theta = 2.0 * np.pi * np.asarray(t, dtype=float) / float(period) + float(phase)
    offsets = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    angles = theta[:, None] + offsets[None, :]
    rel = rho * (np.cos(angles)[..., None] * u[None, None, :] + np.sin(angles)[..., None] * v[None, None, :])
    return np.asarray(center, dtype=float)[:, None, :] + rel


def _bbo_cartwheel_triangle(
    t: np.ndarray,
    center: np.ndarray,
    *,
    arm_length: float,
    center_period: float,
    center_phase: float,
    cartwheel_period: float,
    cartwheel_phase: float,
    plane_inclination: float,
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    alpha = 2.0 * np.pi * t_arr / float(center_period) + float(center_phase)
    normal_inclination = float(plane_inclination)
    n = np.column_stack(
        [
            -np.sin(normal_inclination) * np.cos(alpha),
            -np.sin(normal_inclination) * np.sin(alpha),
            np.full_like(alpha, np.cos(normal_inclination)),
        ]
    )
    u = np.column_stack([-np.sin(alpha), np.cos(alpha), np.zeros_like(alpha)])
    v = np.cross(n, u)
    v = v / np.linalg.norm(v, axis=1)[:, None]
    rho = float(arm_length) / np.sqrt(3.0)
    theta = float(cartwheel_phase) - 2.0 * np.pi * t_arr / float(cartwheel_period)
    offsets = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    angles = theta[:, None] + offsets[None, :]
    rel = rho * (np.cos(angles)[..., None] * u[:, None, :] + np.sin(angles)[..., None] * v[:, None, :])
    return np.asarray(center, dtype=float)[:, None, :] + rel


def build_heliocentric_equal_arm_orbit_arrays(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = 0.0,
    arm_length: float = LISA_NOMINAL_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = LISA_SIMPLE_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
) -> OrbitArrays:
    """Build a self-contained heliocentric rigid equal-arm cartwheel orbit."""

    t_local = _make_local_time_grid(float(duration), float(orbit_dt))
    t_model = t_local + float(time_offset)
    center = _circular_heliocentric_center(
        t_model,
        radius=float(center_radius),
        period=float(center_period),
        phase=float(center_phase),
    )
    x = _bbo_cartwheel_triangle(
        t_model,
        center,
        arm_length=float(arm_length),
        center_period=float(center_period),
        center_phase=float(center_phase),
        cartwheel_period=float(cartwheel_period),
        cartwheel_phase=float(cartwheel_phase),
        plane_inclination=float(plane_inclination),
    )
    return build_orbit_arrays(t_local, x, armlength=float(arm_length))


def build_lisa_simple_orbit_arrays(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = LISA_TRAILING_CENTER_PHASE,
    arm_length: float = LISA_NOMINAL_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = LISA_SIMPLE_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
) -> OrbitArrays:
    """Build the self-contained LISA simple equal-arm cartwheel orbit."""

    return build_heliocentric_equal_arm_orbit_arrays(
        duration=duration,
        orbit_dt=orbit_dt,
        time_offset=time_offset,
        center_radius=center_radius,
        center_period=center_period,
        center_phase=center_phase,
        arm_length=arm_length,
        cartwheel_period=cartwheel_period,
        cartwheel_phase=cartwheel_phase,
        plane_inclination=plane_inclination,
    )


def build_taiji_simple_orbit_arrays(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = TAIJI_AHEAD_CENTER_PHASE,
    arm_length: float = TAIJI_NOMINAL_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = TAIJI_SIMPLE_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
) -> OrbitArrays:
    """Build the self-contained Taiji simple equal-arm cartwheel orbit."""

    return build_heliocentric_equal_arm_orbit_arrays(
        duration=duration,
        orbit_dt=orbit_dt,
        time_offset=time_offset,
        center_radius=center_radius,
        center_period=center_period,
        center_phase=center_phase,
        arm_length=arm_length,
        cartwheel_period=cartwheel_period,
        cartwheel_phase=cartwheel_phase,
        plane_inclination=plane_inclination,
    )


def _interpolate_vector_series(t_source: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Cubic-spline interpolate a ``(nt, ..., 3)`` vector series."""

    flat = values.reshape(len(t_source), -1)
    out = np.empty((len(t_query), flat.shape[-1]), dtype=float)
    for i in range(flat.shape[-1]):
        out[:, i] = interpolate.CubicSpline(t_source, flat[:, i])(t_query)
    return out.reshape((len(t_query),) + values.shape[1:])


def _interpolate_array_series(t_source: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Cubic-spline interpolate an arbitrary array series whose first axis is time."""

    flat = values.reshape(len(t_source), -1)
    out = np.empty((len(t_query), flat.shape[-1]), dtype=float)
    for i in range(flat.shape[-1]):
        out[:, i] = interpolate.CubicSpline(t_source, flat[:, i])(t_query)
    return out.reshape((len(t_query),) + values.shape[1:])


def _make_local_time_grid(duration: float, dt: float) -> np.ndarray:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    n = int(np.ceil(float(duration) / float(dt))) + 1
    t_local = np.arange(n, dtype=float) * float(dt)
    if t_local[-1] < duration:
        t_local = np.concatenate([t_local, np.asarray([float(duration)])])
    else:
        t_local[-1] = float(duration)
    return t_local


def _load_npz_orbit_file(path: str | Path, spec: OrbitSpec) -> OrbitArrays:
    with np.load(path, allow_pickle=False) as data:
        if "t" not in data or "x" not in data:
            raise ValueError("npz orbit files must contain at least 't' and 'x'")
        t = np.asarray(data["t"], dtype=float) * _time_unit_factor(spec.time_unit)
        x = np.asarray(data["x"], dtype=float) * _length_unit_factor(spec.length_unit)
        v = np.asarray(data["v"], dtype=float) * _length_unit_factor(spec.length_unit) / _time_unit_factor(spec.time_unit) if "v" in data else None
        ltt = np.asarray(data["ltt"], dtype=float) * _time_unit_factor(spec.time_unit) if "ltt" in data else None
        links = tuple(int(item) for item in data["links"]) if "links" in data else spec.links
        armlength = float(data["armlength"]) * _length_unit_factor(spec.length_unit) if "armlength" in data else spec.armlength
    return build_orbit_arrays(t, x, links=links, armlength=armlength, v=v, ltt=ltt)


def _load_csv_orbit_file(path: str | Path, spec: OrbitSpec) -> OrbitArrays:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = set(data.dtype.names or ())
    if "t" not in names:
        raise ValueError("csv orbit files must have a 't' column")
    t = np.asarray(data["t"], dtype=float) * _time_unit_factor(spec.time_unit)

    wide_cols = [f"{axis}{sc}" for sc in (1, 2, 3) for axis in ("x", "y", "z")]
    if all(name in names for name in wide_cols):
        x = np.empty((len(t), 3, 3), dtype=float)
        for sc in (1, 2, 3):
            for j, axis in enumerate(("x", "y", "z")):
                x[:, sc - 1, j] = data[f"{axis}{sc}"]
        v_cols = [f"v{axis}{sc}" for sc in (1, 2, 3) for axis in ("x", "y", "z")]
        if all(name in names for name in v_cols):
            v = np.empty_like(x)
            for sc in (1, 2, 3):
                for j, axis in enumerate(("x", "y", "z")):
                    v[:, sc - 1, j] = data[f"v{axis}{sc}"]
        else:
            v = None
    elif {"sc", "x", "y", "z"}.issubset(names):
        unique_t = np.unique(t)
        if len(unique_t) * 3 != len(t):
            raise ValueError("long csv orbit format must contain exactly three spacecraft rows per time")
        x = np.empty((len(unique_t), 3, 3), dtype=float)
        has_v = {"vx", "vy", "vz"}.issubset(names)
        v = np.empty_like(x) if has_v else None
        for i, t_value in enumerate(unique_t):
            mask = t == t_value
            rows = data[mask]
            for row in rows:
                sc = int(row["sc"])
                if sc not in (1, 2, 3):
                    raise ValueError("spacecraft id column 'sc' must contain 1, 2, or 3")
                x[i, sc - 1] = [row["x"], row["y"], row["z"]]
                if has_v:
                    v[i, sc - 1] = [row["vx"], row["vy"], row["vz"]]
        t = unique_t
    else:
        raise ValueError(
            "csv orbit files must use wide columns t,x1,y1,z1,... or long columns t,sc,x,y,z"
        )

    length_factor = _length_unit_factor(spec.length_unit)
    time_factor = _time_unit_factor(spec.time_unit)
    x = x * length_factor
    if v is not None:
        v = v * length_factor / time_factor
    return build_orbit_arrays(t, x, links=spec.links, armlength=spec.armlength, v=v)


def load_orbit_file(path: str | Path, spec: OrbitSpec | dict[str, Any] | None = None) -> OrbitArrays:
    """Load an orbit file into host arrays.

    NPZ files use keys ``t`` and ``x`` with optional ``v``, ``ltt``,
    ``armlength``, and ``links``.  CSV files may be wide
    ``t,x1,y1,z1,...,x3,y3,z3`` or long ``t,sc,x,y,z``.
    """

    orbit_spec = OrbitSpec(base="file") if spec is None else (orbit_spec_from_dict(spec) if isinstance(spec, dict) else spec)
    orbit_path = Path(path)
    fmt = orbit_spec.file_format or orbit_path.suffix.lower().lstrip(".")
    if fmt == "npz":
        return _load_npz_orbit_file(orbit_path, orbit_spec)
    if fmt == "csv":
        return _load_csv_orbit_file(orbit_path, orbit_spec)
    raise ValueError("orbit file format must be 'npz' or 'csv'")


def save_orbit_npz(path: str | Path, arrays: OrbitArrays) -> None:
    """Write a sampled orbit to the simple NPZ interchange format."""

    np.savez(
        path,
        t=arrays.t,
        x=arrays.x,
        v=arrays.v,
        ltt=arrays.ltt,
        armlength=np.asarray(arrays.armlength),
        links=np.asarray(arrays.links, dtype=np.int32),
    )


def _resolve_taiji_triangle_orbit_dir(orbit_dir: str | Path | None) -> Path:
    if orbit_dir is not None:
        path = Path(orbit_dir)
    else:
        path = _default_taiji_triangle_orbit_dir()
    if path is None:
        raise FileNotFoundError(
            "Taiji Triangle orbit data were not found. Download "
            "Triangle-Simulator/OrbitData/TaijiEqualArmOrbit and pass orbit_dir=..., "
            "or set GWDELTA_TAIJI_TRIANGLE_ORBIT_DIR or GWDELTA_ORBIT_DATA_DIR."
        )
    if not path.exists():
        raise FileNotFoundError(
            "Taiji Triangle orbit data were not found. Download "
            "Triangle-Simulator/OrbitData/TaijiEqualArmOrbit and pass orbit_dir=..., "
            f"or set GWDELTA_TAIJI_TRIANGLE_ORBIT_DIR. Checked: {path}"
        )
    missing = [name for name in ("SCP1.dat", "SCP2.dat", "SCP3.dat", "SCV1.dat", "SCV2.dat", "SCV3.dat") if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Taiji Triangle orbit directory is missing files: {', '.join(missing)}")
    return path


def _resolve_taiji_accurate_orbit_dir(orbit_dir: str | Path | None) -> Path:
    if orbit_dir is not None:
        path = Path(orbit_dir)
    else:
        path = _default_taiji_accurate_orbit_dir()
    if path is None:
        raise FileNotFoundError(
            "Taiji accurate orbit data were not found. Download "
            "Triangle-Simulator/OrbitData/MicroSateOrbitEclipticTCB and pass orbit_dir=..., "
            "or set GWDELTA_TAIJI_ACCURATE_ORBIT_DIR or GWDELTA_ORBIT_DATA_DIR."
        )
    orbit_file = path / "MicroSateOrbit.hdf5"
    if not orbit_file.exists():
        raise FileNotFoundError(
            "Taiji accurate orbit data were not found. Download "
            "Triangle-Simulator/OrbitData/MicroSateOrbitEclipticTCB and pass orbit_dir=..., "
            f"or set GWDELTA_TAIJI_ACCURATE_ORBIT_DIR. Checked: {orbit_file}"
        )
    return path


def _load_taiji_triangle_series(
    orbit_dir: str | Path | None = None,
    *,
    data_dt: float = DAY_SI,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Triangle-Simulator Taiji orbit data in SI units."""

    path = _resolve_taiji_triangle_orbit_dir(orbit_dir)
    positions = []
    velocities = []
    load_kwargs = {} if max_rows is None else {"max_rows": int(max_rows)}
    for sc in (1, 2, 3):
        positions.append(np.loadtxt(path / f"SCP{sc}.dat", **load_kwargs) * AU_SI)
        velocities.append(np.loadtxt(path / f"SCV{sc}.dat", **load_kwargs) * AU_SI / DAY_SI)
    x = np.stack(positions, axis=1)
    v = np.stack(velocities, axis=1)
    t = np.arange(x.shape[0], dtype=float) * float(data_dt)
    return t, x, v


def _load_taiji_accurate_hdf5(orbit_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Triangle-Simulator MicroSateOrbit hdf5 data in SI units."""

    try:
        import h5py
    except Exception as exc:  # pragma: no cover - dependency error path
        raise ImportError("make_taiji_accurate_orbits requires h5py") from exc

    path = _resolve_taiji_accurate_orbit_dir(orbit_dir) / "MicroSateOrbit.hdf5"
    with h5py.File(path, "r") as f:
        group = f["tcb"]
        t = np.asarray(group["t"][:], dtype=float)
        x_items = []
        for sc in (1, 2, 3):
            data = group[f"sc_{sc}"][:]
            x_items.append(np.column_stack([data["x"], data["y"], data["z"]]).astype(float))
        x = np.stack(x_items, axis=1)
        ltt_items = []
        for link in DEFAULT_LINKS:
            ltt_items.append(np.asarray(group[f"l_{link}"][:]["tt"], dtype=float))
        ltt = np.stack(ltt_items, axis=1)
    return t, x, ltt


def make_time_shifted_orbits(
    base_orbits: Orbits,
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    rotate_z: float = 0.0,
    force_backend: str | None = None,
) -> SampledOrbits:
    """Return a custom orbit with a shifted/rotated initial detector position.

    The returned orbit has local times starting at zero.  Its spacecraft
    positions at local time ``t`` are sampled from ``base_orbits`` at
    ``t + time_offset`` and then optionally rotated about the ecliptic z axis.
    This is the lightest way to change the default ``t=0`` detector phase while
    keeping the FastLISAResponse API unchanged.
    """

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if orbit_dt <= 0.0:
        raise ValueError("orbit_dt must be positive")

    n = int(np.ceil(float(duration) / float(orbit_dt))) + 1
    t_local = np.arange(n, dtype=float) * float(orbit_dt)
    if t_local[-1] < duration:
        t_local = np.concatenate([t_local, np.asarray([float(duration)])])
    else:
        t_local[-1] = float(duration)

    t_query = t_local + float(time_offset)
    x = _sample_base_positions(base_orbits, t_query)
    x = _rotate_z(x, float(rotate_z))
    return SampledOrbits(
        t_local,
        x,
        links=tuple(base_orbits.LINKS),
        armlength=float(base_orbits.armlength),
        force_backend=force_backend,
    )


def make_taiji_triangle_orbits(
    *,
    orbit_dir: str | Path | None = None,
    duration: float | None = None,
    orbit_dt: float = DAY_SI,
    time_offset: float = 0.0,
    rotate_z: float = 0.0,
    armlength: float = TAIJI_NOMINAL_ARM_M,
    force_backend: str | None = None,
    max_rows: int | None = None,
) -> SampledOrbits:
    """Return Taiji equal-arm orbits from Triangle-Simulator data.

    The public ``OrbitData/TaijiEqualArmOrbit`` samples from Triangle-Simulator
    store positions in AU and velocities in
    AU/day on a one-day grid; this adapter converts them to SI units and then
    creates a FastLISAResponse-compatible ``SampledOrbits`` object.
    """

    t_data, x_data, v_data = _load_taiji_triangle_series(orbit_dir, max_rows=max_rows)
    if time_offset < t_data[0]:
        raise ValueError("time_offset lies before the Taiji orbit data start")
    max_duration = float(t_data[-1] - time_offset)
    if duration is None:
        duration = max_duration
    if duration > max_duration:
        raise ValueError(
            "requested Taiji orbit extends beyond the available Triangle-Simulator data; "
            "reduce duration or time_offset"
        )

    t_local = _make_local_time_grid(float(duration), float(orbit_dt))
    t_query = t_local + float(time_offset)
    x = _interpolate_vector_series(t_data, x_data, t_query)
    v = _interpolate_vector_series(t_data, v_data, t_query)
    x = _rotate_z(x, float(rotate_z))
    v = _rotate_z(v, float(rotate_z))
    return SampledOrbits(
        t_local,
        x,
        v=v,
        links=tuple(DEFAULT_LINKS),
        armlength=float(armlength),
        force_backend=force_backend,
    )


def make_taiji_accurate_orbits(
    *,
    orbit_dir: str | Path | None = None,
    duration: float | None = None,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    rotate_z: float = 0.0,
    armlength: float | None = None,
    force_backend: str | None = None,
) -> SampledOrbits:
    """Return the Triangle-Simulator numerical Taiji orbit.

    This adapter reads ``MicroSateOrbitEclipticTCB/MicroSateOrbit.hdf5``.  The
    hdf5 file provides SI spacecraft positions and per-link TCB light-travel
    times, so the returned orbit preserves the unequal-arm light-time data
    instead of recomputing delays from an equal-arm approximation.
    """

    t_data, x_data, ltt_data = _load_taiji_accurate_hdf5(orbit_dir)
    if time_offset < t_data[0]:
        raise ValueError("time_offset lies before the Taiji accurate orbit data start")
    max_duration = float(t_data[-1] - time_offset)
    if duration is None:
        duration = max_duration
    if duration > max_duration:
        raise ValueError(
            "requested Taiji accurate orbit extends beyond the available MicroSateOrbit data; "
            "reduce duration or time_offset"
        )

    t_local = _make_local_time_grid(float(duration), float(orbit_dt))
    t_query = t_local + float(time_offset)
    x = _interpolate_vector_series(t_data, x_data, t_query)
    ltt = _interpolate_vector_series(t_data, ltt_data, t_query)
    x = _rotate_z(x, float(rotate_z))
    arm = float(np.median(ltt) * C_SI) if armlength is None else float(armlength)
    return SampledOrbits(
        t_local,
        x,
        ltt=ltt,
        links=tuple(DEFAULT_LINKS),
        armlength=arm,
        force_backend=force_backend,
    )


def _resample_arrays(
    arrays: OrbitArrays,
    *,
    duration: float | None,
    orbit_dt: float,
    time_offset: float,
) -> OrbitArrays:
    if duration is None:
        duration = float(arrays.t[-1] - time_offset)
    t_local = _make_local_time_grid(float(duration), float(orbit_dt))
    t_query = t_local + float(time_offset)
    if np.any(t_query < arrays.t[0]) or np.any(t_query > arrays.t[-1]):
        raise ValueError("requested orbit grid lies outside the loaded orbit file")
    x = _interpolate_vector_series(arrays.t, arrays.x, t_query)
    v = _interpolate_vector_series(arrays.t, arrays.v, t_query)
    ltt = _interpolate_array_series(arrays.t, arrays.ltt, t_query)
    return build_orbit_arrays(t_local, x, links=arrays.links, armlength=arrays.armlength, v=v, ltt=ltt)


def _transform_arrays(arrays: OrbitArrays, spec: OrbitSpec) -> OrbitArrays:
    x = _apply_position_transforms(
        arrays.x,
        rotate_z=spec.rotate_z,
        translate=spec.translate,
        scale=spec.scale,
    )
    v = _apply_velocity_transforms(arrays.v, rotate_z=spec.rotate_z, scale=spec.scale)
    ltt = _apply_ltt_scale(arrays.ltt, scale=spec.scale, preserve_ltt=spec.preserve_ltt)
    arm = spec.armlength if spec.armlength is not None else arrays.armlength * abs(float(spec.scale))
    return build_orbit_arrays(arrays.t, x, links=arrays.links, armlength=arm, v=v, ltt=ltt)


def _sampled_from_arrays(arrays: OrbitArrays, *, force_backend: str | None = None) -> SampledOrbits:
    return SampledOrbits(
        arrays.t,
        arrays.x,
        v=arrays.v,
        ltt=arrays.ltt,
        links=arrays.links,
        armlength=arrays.armlength,
        force_backend=force_backend,
    )


def build_tianqin_toy_orbit_arrays(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = 0.0,
    geocentric_radius: float | None = TIANQIN_GEOCENTRIC_RADIUS_M,
    arm_length: float | None = None,
    cartwheel_period: float | None = None,
    cartwheel_phase: float = 0.0,
    normal_lon: float = TIANQIN_J0806_LON,
    normal_lat: float = TIANQIN_J0806_LAT,
) -> OrbitArrays:
    """Build a rigid equal-arm TianQin-like toy orbit.

    The guiding center follows a circular 1 AU Earth orbit.  The triangular
    constellation is geocentric, rigid, and has a fixed inertial plane whose
    normal points to the verification binary RX J0806.3+1527 by default.
    """

    if geocentric_radius is None and arm_length is None:
        geocentric_radius = TIANQIN_GEOCENTRIC_RADIUS_M
    if arm_length is None:
        arm_length = np.sqrt(3.0) * float(geocentric_radius)
    if geocentric_radius is None:
        geocentric_radius = float(arm_length) / np.sqrt(3.0)
    if cartwheel_period is None:
        cartwheel_period = 2.0 * np.pi * np.sqrt(float(geocentric_radius) ** 3 / EARTH_MU_SI)

    t_local = _make_local_time_grid(float(duration), float(orbit_dt))
    t_model = t_local + float(time_offset)
    center = _circular_heliocentric_center(
        t_model,
        radius=float(center_radius),
        period=float(center_period),
        phase=float(center_phase),
    )
    normal = _unit_from_ecliptic(float(normal_lon), float(normal_lat))
    x = _fixed_normal_rigid_triangle(
        t_model,
        center,
        arm_length=float(arm_length),
        normal=normal,
        period=float(cartwheel_period),
        phase=float(cartwheel_phase),
    )
    return build_orbit_arrays(t_local, x, armlength=float(arm_length))


def build_bbo_stage1_toy_orbit_arrays(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = BBO_STAGE1_CENTER_PHASE,
    arm_length: float = BBO_STAGE1_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = BBO_STAGE1_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
) -> OrbitArrays:
    """Build a rigid equal-arm BBO Stage-I-like toy orbit.

    The default follows the public Stage-I numbers, ``L=5e7 m`` at 1 AU with a
    one-year constellation rotation.  The 60 degree inclination is the angle
    between the detector-plane normal and the ecliptic normal, matching the
    LISA-like cartwheel convention; change it in the JSON config if a different
    convention is desired.
    """

    t_local = _make_local_time_grid(float(duration), float(orbit_dt))
    t_model = t_local + float(time_offset)
    center = _circular_heliocentric_center(
        t_model,
        radius=float(center_radius),
        period=float(center_period),
        phase=float(center_phase),
    )
    x = _bbo_cartwheel_triangle(
        t_model,
        center,
        arm_length=float(arm_length),
        center_period=float(center_period),
        center_phase=float(center_phase),
        cartwheel_period=float(cartwheel_period),
        cartwheel_phase=float(cartwheel_phase),
        plane_inclination=float(plane_inclination),
    )
    return build_orbit_arrays(t_local, x, armlength=float(arm_length))


def make_lisa_simple_orbits(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = LISA_TRAILING_CENTER_PHASE,
    arm_length: float = LISA_NOMINAL_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = LISA_SIMPLE_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
    force_backend: str | None = None,
) -> SampledOrbits:
    arrays = build_lisa_simple_orbit_arrays(
        duration=duration,
        orbit_dt=orbit_dt,
        time_offset=time_offset,
        center_radius=center_radius,
        center_period=center_period,
        center_phase=center_phase,
        arm_length=arm_length,
        cartwheel_period=cartwheel_period,
        cartwheel_phase=cartwheel_phase,
        plane_inclination=plane_inclination,
    )
    return _sampled_from_arrays(arrays, force_backend=force_backend)


def make_taiji_simple_orbits(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = TAIJI_AHEAD_CENTER_PHASE,
    arm_length: float = TAIJI_NOMINAL_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = TAIJI_SIMPLE_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
    force_backend: str | None = None,
) -> SampledOrbits:
    arrays = build_taiji_simple_orbit_arrays(
        duration=duration,
        orbit_dt=orbit_dt,
        time_offset=time_offset,
        center_radius=center_radius,
        center_period=center_period,
        center_phase=center_phase,
        arm_length=arm_length,
        cartwheel_period=cartwheel_period,
        cartwheel_phase=cartwheel_phase,
        plane_inclination=plane_inclination,
    )
    return _sampled_from_arrays(arrays, force_backend=force_backend)


def make_tianqin_toy_orbits(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = 0.0,
    geocentric_radius: float | None = TIANQIN_GEOCENTRIC_RADIUS_M,
    arm_length: float | None = None,
    cartwheel_period: float | None = None,
    cartwheel_phase: float = 0.0,
    normal_lon: float = TIANQIN_J0806_LON,
    normal_lat: float = TIANQIN_J0806_LAT,
    force_backend: str | None = None,
) -> SampledOrbits:
    arrays = build_tianqin_toy_orbit_arrays(
        duration=duration,
        orbit_dt=orbit_dt,
        time_offset=time_offset,
        center_radius=center_radius,
        center_period=center_period,
        center_phase=center_phase,
        geocentric_radius=geocentric_radius,
        arm_length=arm_length,
        cartwheel_period=cartwheel_period,
        cartwheel_phase=cartwheel_phase,
        normal_lon=normal_lon,
        normal_lat=normal_lat,
    )
    return _sampled_from_arrays(arrays, force_backend=force_backend)


def make_bbo_stage1_toy_orbits(
    *,
    duration: float,
    orbit_dt: float = LINEAR_INTERP_TIMESTEP,
    time_offset: float = 0.0,
    center_radius: float = AU_SI,
    center_period: float = SIDEREAL_YEAR_S,
    center_phase: float = BBO_STAGE1_CENTER_PHASE,
    arm_length: float = BBO_STAGE1_ARM_M,
    cartwheel_period: float = SIDEREAL_YEAR_S,
    cartwheel_phase: float = BBO_STAGE1_CARTWHEEL_PHASE,
    plane_inclination: float = BBO_STAGE1_PLANE_INCLINATION,
    force_backend: str | None = None,
) -> SampledOrbits:
    arrays = build_bbo_stage1_toy_orbit_arrays(
        duration=duration,
        orbit_dt=orbit_dt,
        time_offset=time_offset,
        center_radius=center_radius,
        center_period=center_period,
        center_phase=center_phase,
        arm_length=arm_length,
        cartwheel_period=cartwheel_period,
        cartwheel_phase=cartwheel_phase,
        plane_inclination=plane_inclination,
    )
    return _sampled_from_arrays(arrays, force_backend=force_backend)


def _make_native_lisatools_orbits(base: str, *, force_backend: str | None = None):
    if not _HAS_LISATOOLS:
        raise ImportError("lisaanalysistools is required for built-in ESA/equal-arm orbits")
    from lisatools.detector import ESAOrbits, EqualArmlengthOrbits

    if base == "esa":
        return ESAOrbits(force_backend=force_backend)
    if base in ("equal-armlength", "equal_armlength", "equal"):
        return EqualArmlengthOrbits(force_backend=force_backend)
    raise ValueError(f"unknown native lisatools orbit base {base!r}")


def make_orbits_from_spec(
    spec: OrbitSpec | dict[str, Any] | str | Path,
    *,
    duration: float | None = None,
    force_backend: str | None = None,
) -> Orbits:
    """Build a FastLISAResponse-compatible orbit object from a user spec.

    ``spec`` may be an ``OrbitSpec``, a plain dictionary, or a JSON path.  The
    optional ``duration`` overrides the spec duration; this is useful for CLI
    benchmark scripts whose duration is derived from ``n_samples * dt``.
    """

    if isinstance(spec, (str, Path)):
        orbit_spec = load_orbit_spec(spec)
    elif isinstance(spec, dict):
        orbit_spec = orbit_spec_from_dict(spec)
    else:
        orbit_spec = spec
    if duration is not None:
        orbit_spec = OrbitSpec(**{**orbit_spec.__dict__, "duration": float(duration)})

    base = orbit_spec.base.strip().lower()
    target_center_phase = _default_center_phase_for_base(base, orbit_spec)
    if base in ("esa", "equal-armlength", "equal_armlength", "equal"):
        native = _make_native_lisatools_orbits(base, force_backend=force_backend)
        if target_center_phase is None and not _has_sampling_or_transform(orbit_spec):
            return native
        if orbit_spec.duration is None:
            raise ValueError("duration is required when transforming or resampling a native lisatools orbit")
        t_local = _make_local_time_grid(orbit_spec.duration, orbit_spec.orbit_dt)
        t_query = t_local + orbit_spec.time_offset
        arrays = _sample_orbit_arrays_from_lisatools(
            native,
            t_query,
            t_local=t_local,
        )
        arrays = _align_arrays_center_phase(arrays, target_center_phase)
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base in ("lisa-simple", "lisa_simple"):
        if orbit_spec.duration is None:
            raise ValueError("duration is required for lisa-simple orbits")
        arrays = build_lisa_simple_orbit_arrays(
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt,
            time_offset=orbit_spec.time_offset,
            center_radius=AU_SI if orbit_spec.center_radius is None else orbit_spec.center_radius,
            center_period=SIDEREAL_YEAR_S if orbit_spec.center_period is None else orbit_spec.center_period,
            center_phase=(
                LISA_TRAILING_CENTER_PHASE
                if orbit_spec.center_phase is None
                else orbit_spec.center_phase
            ),
            arm_length=LISA_NOMINAL_ARM_M if orbit_spec.armlength is None else orbit_spec.armlength,
            cartwheel_period=(
                SIDEREAL_YEAR_S
                if orbit_spec.cartwheel_period is None
                else orbit_spec.cartwheel_period
            ),
            cartwheel_phase=(
                LISA_SIMPLE_CARTWHEEL_PHASE
                if orbit_spec.cartwheel_phase is None
                else orbit_spec.cartwheel_phase
            ),
            plane_inclination=(
                BBO_STAGE1_PLANE_INCLINATION
                if orbit_spec.plane_inclination is None
                else orbit_spec.plane_inclination
            ),
        )
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base in ("taiji-simple", "taiji_simple"):
        if orbit_spec.duration is None:
            raise ValueError("duration is required for taiji-simple orbits")
        arrays = build_taiji_simple_orbit_arrays(
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt,
            time_offset=orbit_spec.time_offset,
            center_radius=AU_SI if orbit_spec.center_radius is None else orbit_spec.center_radius,
            center_period=SIDEREAL_YEAR_S if orbit_spec.center_period is None else orbit_spec.center_period,
            center_phase=(
                TAIJI_AHEAD_CENTER_PHASE
                if orbit_spec.center_phase is None
                else orbit_spec.center_phase
            ),
            arm_length=TAIJI_NOMINAL_ARM_M if orbit_spec.armlength is None else orbit_spec.armlength,
            cartwheel_period=(
                SIDEREAL_YEAR_S
                if orbit_spec.cartwheel_period is None
                else orbit_spec.cartwheel_period
            ),
            cartwheel_phase=(
                TAIJI_SIMPLE_CARTWHEEL_PHASE
                if orbit_spec.cartwheel_phase is None
                else orbit_spec.cartwheel_phase
            ),
            plane_inclination=(
                BBO_STAGE1_PLANE_INCLINATION
                if orbit_spec.plane_inclination is None
                else orbit_spec.plane_inclination
            ),
        )
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base == "taiji-triangle":
        t_data, x_data, v_data = _load_taiji_triangle_series(orbit_spec.orbit_dir, max_rows=orbit_spec.max_rows)
        arrays = build_orbit_arrays(t_data, x_data, links=orbit_spec.links, armlength=TAIJI_NOMINAL_ARM_M, v=v_data)
        arrays = _resample_arrays(
            arrays,
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt if orbit_spec.orbit_dt is not None else DAY_SI,
            time_offset=orbit_spec.time_offset,
        )
        arrays = _align_arrays_center_phase(arrays, target_center_phase)
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base == "taiji-accurate":
        t_data, x_data, ltt_data = _load_taiji_accurate_hdf5(orbit_spec.orbit_dir)
        arm = float(np.median(ltt_data) * C_SI) if orbit_spec.armlength is None else orbit_spec.armlength
        arrays = build_orbit_arrays(t_data, x_data, links=orbit_spec.links, armlength=arm, ltt=ltt_data)
        arrays = _resample_arrays(
            arrays,
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt,
            time_offset=orbit_spec.time_offset,
        )
        arrays = _align_arrays_center_phase(arrays, target_center_phase)
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base in ("tianqin-toy", "tianqin", "tq"):
        if orbit_spec.duration is None:
            raise ValueError("duration is required for tianqin-toy orbits")
        arrays = build_tianqin_toy_orbit_arrays(
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt,
            time_offset=orbit_spec.time_offset,
            center_radius=AU_SI if orbit_spec.center_radius is None else orbit_spec.center_radius,
            center_period=SIDEREAL_YEAR_S if orbit_spec.center_period is None else orbit_spec.center_period,
            center_phase=0.0 if orbit_spec.center_phase is None else orbit_spec.center_phase,
            geocentric_radius=orbit_spec.geocentric_radius,
            arm_length=orbit_spec.armlength,
            cartwheel_period=orbit_spec.cartwheel_period,
            cartwheel_phase=0.0 if orbit_spec.cartwheel_phase is None else orbit_spec.cartwheel_phase,
            normal_lon=TIANQIN_J0806_LON if orbit_spec.normal_lon is None else orbit_spec.normal_lon,
            normal_lat=TIANQIN_J0806_LAT if orbit_spec.normal_lat is None else orbit_spec.normal_lat,
        )
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base in ("bbo-stage1-toy", "bbo-stage1", "bbo"):
        if orbit_spec.duration is None:
            raise ValueError("duration is required for bbo-stage1-toy orbits")
        arrays = build_bbo_stage1_toy_orbit_arrays(
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt,
            time_offset=orbit_spec.time_offset,
            center_radius=AU_SI if orbit_spec.center_radius is None else orbit_spec.center_radius,
            center_period=SIDEREAL_YEAR_S if orbit_spec.center_period is None else orbit_spec.center_period,
            center_phase=BBO_STAGE1_CENTER_PHASE if orbit_spec.center_phase is None else orbit_spec.center_phase,
            arm_length=BBO_STAGE1_ARM_M if orbit_spec.armlength is None else orbit_spec.armlength,
            cartwheel_period=SIDEREAL_YEAR_S if orbit_spec.cartwheel_period is None else orbit_spec.cartwheel_period,
            cartwheel_phase=(
                BBO_STAGE1_CARTWHEEL_PHASE
                if orbit_spec.cartwheel_phase is None
                else orbit_spec.cartwheel_phase
            ),
            plane_inclination=(
                BBO_STAGE1_PLANE_INCLINATION
                if orbit_spec.plane_inclination is None
                else orbit_spec.plane_inclination
            ),
        )
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    if base == "file":
        if orbit_spec.path is None:
            raise ValueError("file orbit specs require a path")
        arrays = load_orbit_file(orbit_spec.path, orbit_spec)
        arrays = _resample_arrays(
            arrays,
            duration=orbit_spec.duration,
            orbit_dt=orbit_spec.orbit_dt,
            time_offset=orbit_spec.time_offset,
        )
        arrays = _align_arrays_center_phase(arrays, orbit_spec.center_phase)
        arrays = _transform_arrays(arrays, orbit_spec)
        return _sampled_from_arrays(arrays, force_backend=force_backend)

    raise ValueError(
        "orbit spec base must be esa, equal-armlength, lisa-simple, taiji-simple, taiji-triangle, "
        "taiji-accurate, tianqin-toy, bbo-stage1-toy, or file"
    )


def make_orbits_from_config(
    path: str | Path,
    *,
    duration: float | None = None,
    force_backend: str | None = None,
) -> Orbits:
    """Build an orbit object from a JSON config file."""

    return make_orbits_from_spec(load_orbit_spec(path), duration=duration, force_backend=force_backend)
