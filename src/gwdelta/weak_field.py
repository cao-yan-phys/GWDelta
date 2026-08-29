"""Finite-arm one-way responses to general linear metric perturbations.

The module computes a perturbative fractional-frequency residual relative to a
supplied background orbit. Link ``ij`` is received at spacecraft ``i`` after
emission from spacecraft ``j``.  The leading fixed-ephemeris observable is

    y = psi_e - psi_r + integral(dt * partial_t P),

where ``P = psi - n.xi - 0.5 * n.H.n`` is evaluated on the zeroth-order photon
chord. An optional metric-induced endpoint-velocity term can be added. Only
velocity-independent terms are retained. In
geometric units, its leading nonrelativistic test-mass equation is
``d(delta V_i)/dt = -partial_i psi - partial_t xi_i`` on the supplied
background trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import CubicSpline

from .array_backend import ArrayBackend, infer_backend_from_array, select_array_backend
from .cuda_runtime import ensure_cuda_dll_directories
from .fd_response import (
    C_SI,
    DEFAULT_LINKS,
    POLARIZATION_NAMES,
    polarization_tensors,
    sky_basis,
)


G_SI = 6.67430e-11


@dataclass(frozen=True)
class MetricValues:
    """Weak-field components on a common broadcast grid."""

    psi: Any
    xi: Any
    h: Any


@runtime_checkable
class WeakMetricField(Protocol):
    """Vectorized weak metric in the GWDelta coordinate convention."""

    def metric(self, t_s, x_m) -> MetricValues:
        """Return ``psi``, ``xi_i``, and ``h_ij`` at ``(t_s, x_m)``."""

    def time_derivative(self, t_s, x_m) -> MetricValues:
        """Return physical-time derivatives of all metric components."""


@runtime_checkable
class MonochromaticWeakMetricField(WeakMetricField, Protocol):
    """Weak metric with a declared complex-phasor time convention."""

    angular_frequency: float
    phasor_sign: int

    def complex_amplitude(self, x_m) -> MetricValues:
        """Return spatial complex amplitudes of the metric components."""


@dataclass(frozen=True)
class LinkGeometry:
    """Emission/reception geometry for all directed links."""

    t_reception_s: np.ndarray
    t_emission_s: np.ndarray
    x_reception_m: np.ndarray
    x_emission_m: np.ndarray
    v_reception_m_s: np.ndarray
    v_emission_m_s: np.ndarray
    direction: np.ndarray
    light_time_s: np.ndarray
    chord_length_m: np.ndarray
    links: tuple[int, ...]
    max_null_mismatch: float
    max_emission_extrapolation_s: float


@dataclass
class LinkSignalResult:
    """Time-domain one-way metric-perturbation residuals."""

    t: np.ndarray
    links: tuple[int, ...]
    direct: Any
    worldline: Any
    total: Any
    geometry: LinkGeometry
    metadata: dict[str, Any]

    def as_numpy(self) -> dict[str, np.ndarray]:
        backend = infer_backend_from_array(self.total)
        return {
            "t": np.asarray(self.t),
            "direct": backend.asnumpy(self.direct),
            "worldline": backend.asnumpy(self.worldline),
            "total": backend.asnumpy(self.total),
        }


@dataclass(frozen=True)
class TestMassPerturbation:
    """Leading motion induced along sampled background worldlines."""

    t: Any
    acceleration_m_s2: Any
    delta_velocity_m_s: Any
    delta_position_m: Any

    def as_numpy(self) -> dict[str, np.ndarray]:
        backend = infer_backend_from_array(self.delta_velocity_m_s)
        return {
            "t": backend.asnumpy(self.t),
            "acceleration_m_s2": backend.asnumpy(self.acceleration_m_s2),
            "delta_velocity_m_s": backend.asnumpy(self.delta_velocity_m_s),
            "delta_position_m": backend.asnumpy(self.delta_position_m),
        }


class PlaneWavePolarizationField:
    """Sampled six-polarization null plane wave in synchronous gauge.

    ``h_polarizations`` maps one or more names from ``POLARIZATION_NAMES`` to
    real strain samples on the uniform SSB time grid ``time_s``. The field
    evaluates the retarded argument at the requested SSB position, constructs
    the polarization tensors internally, and supports NumPy or CuPy execution.
    """

    def __init__(
        self,
        time_s,
        h_polarizations: Mapping[str, Any],
        *,
        lam: float,
        beta: float,
        reference_position_m=(0.0, 0.0, 0.0),
    ) -> None:
        time_host = np.asarray(_to_numpy(time_s), dtype=float)
        if time_host.ndim != 1 or len(time_host) < 2:
            raise ValueError("time_s must be a one-dimensional array with at least two samples")
        if not np.all(np.isfinite(time_host)):
            raise ValueError("time_s must be finite")
        steps = np.diff(time_host)
        dt_s = float(np.median(steps))
        if dt_s <= 0.0 or not np.allclose(
            steps, dt_s, rtol=1.0e-11, atol=max(1.0e-12, abs(dt_s) * 1.0e-12)
        ):
            raise ValueError("time_s must be strictly increasing and uniform")
        if not isinstance(h_polarizations, Mapping) or not h_polarizations:
            raise ValueError("h_polarizations must be a nonempty mapping")

        unknown = [name for name in h_polarizations if name not in POLARIZATION_NAMES]
        if unknown:
            raise ValueError(f"unknown polarization names: {unknown}")
        names = tuple(name for name in POLARIZATION_NAMES if name in h_polarizations)
        values: dict[str, Any] = {}
        for name in names:
            backend = infer_backend_from_array(h_polarizations[name])
            value = backend.xp.asarray(h_polarizations[name], dtype=backend.xp.float64)
            if value.ndim != 1 or value.shape[0] != len(time_host):
                raise ValueError(
                    f"{name} samples must have shape ({len(time_host)},); got {value.shape}"
                )
            if bool(_to_numpy(backend.xp.any(~backend.xp.isfinite(value)))):
                raise ValueError(f"{name} samples must be finite")
            values[name] = value

        reference_position = np.asarray(reference_position_m, dtype=float)
        if reference_position.shape != (3,) or not np.all(np.isfinite(reference_position)):
            raise ValueError("reference_position_m must be a finite three-vector")

        if not np.isfinite(lam) or not np.isfinite(beta):
            raise ValueError("lam and beta must be finite")
        propagation, _a, _b = sky_basis(float(lam), float(beta))
        tensors = polarization_tensors(float(lam), float(beta))
        self.time_s = time_s
        self.h_polarizations = values
        self.lam = float(lam)
        self.beta = float(beta)
        self.reference_position_m = reference_position
        self.polarization_names = names
        self._time_host = time_host
        self._dt_s = dt_s
        self._propagation = propagation
        self._tensors = np.stack([tensors[name] for name in names])
        self._backend_cache: dict[str, tuple[Any, Any, Any, Any, Any, Any]] = {}

    @property
    def time_support_s(self) -> tuple[float, float]:
        """Inclusive SSB-time support of the supplied strain samples."""

        return float(self._time_host[0]), float(self._time_host[-1])

    @staticmethod
    def _uniform_derivative(values, dt_s: float, xp):
        derivative = xp.empty_like(values)
        if values.shape[1] == 2:
            slope = (values[:, 1] - values[:, 0]) / dt_s
            derivative[:, 0] = slope
            derivative[:, 1] = slope
            return derivative
        derivative[:, 0] = (-3.0 * values[:, 0] + 4.0 * values[:, 1] - values[:, 2]) / (
            2.0 * dt_s
        )
        derivative[:, -1] = (
            3.0 * values[:, -1] - 4.0 * values[:, -2] + values[:, -3]
        ) / (2.0 * dt_s)
        derivative[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * dt_s)
        return derivative

    def _arrays_for(self, xp):
        key = xp.__name__
        cached = self._backend_cache.get(key)
        if cached is not None:
            return cached
        time = xp.asarray(self.time_s, dtype=xp.float64)
        values = xp.stack(
            [xp.asarray(self.h_polarizations[name], dtype=xp.float64) for name in self.polarization_names]
        )
        derivatives = self._uniform_derivative(values, self._dt_s, xp)
        tensors = xp.asarray(self._tensors, dtype=xp.float64)
        propagation = xp.asarray(self._propagation, dtype=xp.float64)
        reference_position = xp.asarray(self.reference_position_m, dtype=xp.float64)
        cached = (time, values, derivatives, tensors, propagation, reference_position)
        self._backend_cache[key] = cached
        return cached

    def _interpolate(self, query, xp):
        time, values, derivatives, tensors, propagation, reference_position = self._arrays_for(xp)
        if bool(_to_numpy(xp.any(~xp.isfinite(query)))):
            raise ValueError("retarded plane-wave times must be finite")
        query_min = float(_to_numpy(xp.min(query)))
        query_max = float(_to_numpy(xp.max(query)))
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0,
            abs(query_min),
            abs(query_max),
            abs(self._time_host[0]),
            abs(self._time_host[-1]),
        )
        if query_min < self._time_host[0] - tolerance or query_max > self._time_host[-1] + tolerance:
            raise ValueError(
                "plane-wave time support does not cover the requested retarded times; "
                "extend time_s or set reference_position_m"
            )

        index = xp.clip(xp.searchsorted(time, query, side="right") - 1, 0, len(time) - 2)
        left_time = time[index]
        interval = time[index + 1] - left_time
        coordinate = (query - left_time) / interval
        coordinate2 = coordinate * coordinate
        coordinate3 = coordinate2 * coordinate

        value_left = values[:, index]
        value_right = values[:, index + 1]
        derivative_left = derivatives[:, index]
        derivative_right = derivatives[:, index + 1]
        interval_expanded = interval[xp.newaxis, ...]
        value = (
            value_left * (2.0 * coordinate3 - 3.0 * coordinate2 + 1.0)[xp.newaxis, ...]
            + derivative_left
            * interval_expanded
            * (coordinate3 - 2.0 * coordinate2 + coordinate)[xp.newaxis, ...]
            + value_right * (-2.0 * coordinate3 + 3.0 * coordinate2)[xp.newaxis, ...]
            + derivative_right
            * interval_expanded
            * (coordinate3 - coordinate2)[xp.newaxis, ...]
        )
        derivative = (
            value_left * (6.0 * coordinate2 - 6.0 * coordinate)[xp.newaxis, ...]
            + derivative_left
            * interval_expanded
            * (3.0 * coordinate2 - 4.0 * coordinate + 1.0)[xp.newaxis, ...]
            + value_right * (-6.0 * coordinate2 + 6.0 * coordinate)[xp.newaxis, ...]
            + derivative_right
            * interval_expanded
            * (3.0 * coordinate2 - 2.0 * coordinate)[xp.newaxis, ...]
        ) / interval_expanded
        return (
            xp.moveaxis(value, 0, -1),
            xp.moveaxis(derivative, 0, -1),
            tensors,
            propagation,
            reference_position,
        )

    @staticmethod
    def _broadcast_time_and_position(t_s, x_m, xp):
        time = xp.asarray(t_s, dtype=xp.float64)
        position = xp.asarray(x_m, dtype=xp.float64)
        if position.shape[-1:] != (3,):
            raise ValueError("x_m must have final dimension 3")
        try:
            time = xp.broadcast_to(time, position.shape[:-1])
        except ValueError as exc:
            raise ValueError("t_s must broadcast to x_m.shape[:-1]") from exc
        return time, position

    def _metric_values(self, t_s, x_m, *, derivative: bool) -> MetricValues:
        backend = infer_backend_from_array(x_m)
        xp = backend.xp
        time, position = self._broadcast_time_and_position(t_s, x_m, xp)
        _support_time, _values, _derivatives, _tensors, propagation, reference_position = self._arrays_for(xp)
        retarded_time = time - xp.einsum(
            "...i,i->...", position - reference_position, propagation
        ) / C_SI
        samples, sample_derivative, tensors, _propagation, _reference_position = self._interpolate(
            retarded_time, xp
        )
        amplitude = sample_derivative if derivative else samples
        h = xp.einsum("...a,aij->...ij", amplitude, tensors)
        zeros = xp.zeros(retarded_time.shape, dtype=xp.float64)
        return MetricValues(
            psi=zeros,
            xi=xp.zeros(retarded_time.shape + (3,), dtype=xp.float64),
            h=h,
        )

    def metric(self, t_s, x_m) -> MetricValues:
        """Return the synchronous-gauge metric perturbation."""

        return self._metric_values(t_s, x_m, derivative=False)

    def time_derivative(self, t_s, x_m) -> MetricValues:
        """Return the physical-time derivative of the metric perturbation."""

        return self._metric_values(t_s, x_m, derivative=True)


def _integrate_sampled_acceleration(
    t_s,
    background_position_m,
    acceleration_m_s2,
    *,
    initial_delta_velocity_m_s=None,
    initial_delta_position_m=None,
) -> TestMassPerturbation:
    """Integrate a leading acceleration along sampled background worldlines."""

    xp = _xp_for(background_position_m)
    t = xp.asarray(t_s, dtype=xp.float64)
    position = xp.asarray(background_position_m, dtype=xp.float64)
    acceleration = xp.asarray(acceleration_m_s2, dtype=xp.float64)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t_s must be a one-dimensional array with at least two samples")
    if position.shape[0] != len(t) or position.shape[-1:] != (3,):
        raise ValueError("background_position_m must have shape (len(t_s), ..., 3)")
    if acceleration.shape != position.shape:
        raise ValueError("acceleration_m_s2 must match background_position_m")
    dt = xp.diff(t)
    if bool(_to_numpy(xp.any(dt <= 0.0))):
        raise ValueError("t_s must be strictly increasing")

    initial_shape = position.shape[1:]

    def initial_value(value, name: str):
        if value is None:
            return xp.zeros(initial_shape, dtype=xp.float64)
        array = xp.asarray(value, dtype=xp.float64)
        try:
            return xp.broadcast_to(array, initial_shape).copy()
        except ValueError as exc:
            raise ValueError(f"{name} must broadcast to shape {initial_shape}") from exc

    delta_v0 = initial_value(
        initial_delta_velocity_m_s, "initial_delta_velocity_m_s"
    )
    delta_x0 = initial_value(initial_delta_position_m, "initial_delta_position_m")
    dt_shape = (len(dt),) + (1,) * (acceleration.ndim - 1)
    velocity_steps = 0.5 * (acceleration[1:] + acceleration[:-1]) * dt.reshape(
        dt_shape
    )
    delta_velocity = xp.concatenate(
        [
            delta_v0[xp.newaxis, ...],
            delta_v0[xp.newaxis, ...] + xp.cumsum(velocity_steps, axis=0),
        ],
        axis=0,
    )
    position_steps = 0.5 * (delta_velocity[1:] + delta_velocity[:-1]) * dt.reshape(
        dt_shape
    )
    delta_position = xp.concatenate(
        [
            delta_x0[xp.newaxis, ...],
            delta_x0[xp.newaxis, ...] + xp.cumsum(position_steps, axis=0),
        ],
        axis=0,
    )
    return TestMassPerturbation(
        t=t,
        acceleration_m_s2=acceleration,
        delta_velocity_m_s=delta_velocity,
        delta_position_m=delta_position,
    )


@dataclass(frozen=True)
class MonochromaticLinkResult:
    """Frozen-geometry complex one-way amplitudes."""

    angular_frequency: float
    reference_time_s: float
    links: tuple[int, ...]
    direct_amplitude: Any
    geometry: LinkGeometry
    metadata: dict[str, Any]

    def as_numpy(self) -> np.ndarray:
        return infer_backend_from_array(self.direct_amplitude).asnumpy(
            self.direct_amplitude
        )


def _to_numpy(value) -> np.ndarray:
    return infer_backend_from_array(value).asnumpy(value)


def _backend_from_name(force_backend: str | None) -> ArrayBackend:
    if force_backend is None:
        return select_array_backend(prefer_gpu=False, force="cpu")
    key = str(force_backend).strip().lower()
    if key in {"cpu", "numpy"}:
        return select_array_backend(prefer_gpu=False, force="cpu")
    if "cuda" in key or key in {"cupy", "gpu"}:
        ensure_cuda_dll_directories()
        return select_array_backend(prefer_gpu=True, force="cuda")
    raise ValueError("force_backend must be CPU/NumPy or a CUDA/CuPy backend name")


def _sample_series(t_base: np.ndarray, values: np.ndarray, query) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    query_arr = np.asarray(query, dtype=float)
    flat = values.reshape(len(t_base), -1)
    sampled = np.empty(query_arr.shape + (flat.shape[1],), dtype=float)
    for index in range(flat.shape[1]):
        sampled[..., index] = CubicSpline(t_base, flat[:, index], extrapolate=True)(
            query_arr
        )
    return sampled.reshape(query_arr.shape + values.shape[1:])


def _spacecraft_indices(links: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    receivers = np.asarray([int(str(link)[0]) - 1 for link in links], dtype=int)
    emitters = np.asarray([int(str(link)[1]) - 1 for link in links], dtype=int)
    return receivers, emitters


def build_link_geometry(orbits, t_reception_s) -> LinkGeometry:
    """Sample retarded endpoint geometry from any GWDelta-compatible orbit."""

    t = np.asarray(t_reception_s, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("t_reception_s must be a non-empty one-dimensional array")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t_reception_s must be strictly increasing")

    t_base = np.asarray(_to_numpy(orbits.t_base), dtype=float)
    x_base = np.asarray(_to_numpy(orbits.x_base), dtype=float)
    v_base = np.asarray(_to_numpy(orbits.v_base), dtype=float)
    ltt_base = np.asarray(_to_numpy(orbits.ltt_base), dtype=float)
    links = tuple(int(link) for link in orbits.LINKS)
    if links != tuple(DEFAULT_LINKS):
        raise ValueError(
            f"orbit link order must be {tuple(DEFAULT_LINKS)}; got {links}"
        )
    if t[0] < t_base[0] or t[-1] > t_base[-1]:
        raise ValueError(
            "reception-time grid lies outside the available orbit interval"
        )

    receivers, emitters = _spacecraft_indices(links)
    x_at_reception = _sample_series(t_base, x_base, t)
    v_at_reception = _sample_series(t_base, v_base, t)
    ltt_at_reception = _sample_series(t_base, ltt_base, t)
    if ltt_at_reception.shape != (len(t), len(links)):
        raise ValueError("orbit light-time array has an unexpected shape")
    if np.any(ltt_at_reception <= 0.0):
        raise ValueError("orbit produced a non-positive one-way light time")

    light_time = ltt_at_reception.T
    t_emission = t[np.newaxis, :] - light_time
    x_reception = np.moveaxis(x_at_reception[:, receivers, :], 0, 1)
    v_reception = np.moveaxis(v_at_reception[:, receivers, :], 0, 1)
    x_emission = np.empty_like(x_reception)
    v_emission = np.empty_like(v_reception)
    for link_index, spacecraft in enumerate(emitters):
        x_emission[link_index] = _sample_series(
            t_base, x_base[:, spacecraft, :], t_emission[link_index]
        )
        v_emission[link_index] = _sample_series(
            t_base, v_base[:, spacecraft, :], t_emission[link_index]
        )

    chord = x_reception - x_emission
    chord_length = np.linalg.norm(chord, axis=-1)
    if np.any(chord_length <= 0.0):
        raise ValueError("orbit produced a zero-length photon chord")
    direction = chord / chord_length[..., np.newaxis]
    null_mismatch = np.abs(chord_length / (C_SI * light_time) - 1.0)
    emission_extrapolation = max(0.0, float(t_base[0] - np.min(t_emission)))
    allowed_extrapolation = max(
        2.0 * float(np.max(light_time)),
        float(np.median(np.diff(t_base))) if len(t_base) > 1 else 0.0,
    )
    if emission_extrapolation > allowed_extrapolation:
        raise ValueError(
            "retarded emission events require orbit extrapolation beyond one orbit interval"
        )

    return LinkGeometry(
        t_reception_s=t,
        t_emission_s=t_emission,
        x_reception_m=x_reception,
        x_emission_m=x_emission,
        v_reception_m_s=v_reception,
        v_emission_m_s=v_emission,
        direction=direction,
        light_time_s=light_time,
        chord_length_m=chord_length,
        links=links,
        max_null_mismatch=float(np.max(null_mismatch)),
        max_emission_extrapolation_s=emission_extrapolation,
    )


def _validate_metric_values(values: MetricValues, shape: tuple[int, ...]) -> None:
    if values.psi.shape != shape:
        raise ValueError(f"field psi must have shape {shape}; got {values.psi.shape}")
    if values.xi.shape != shape + (3,):
        raise ValueError(
            f"field xi must have shape {shape + (3,)}; got {values.xi.shape}"
        )
    if values.h.shape != shape + (3, 3):
        raise ValueError(
            f"field h must have shape {shape + (3, 3)}; got {values.h.shape}"
        )


def _propagation_kernel(values: MetricValues, direction, xp):
    n_dot_xi = xp.einsum("...i,...i->...", direction, values.xi)
    n_h_n = xp.einsum("...i,...ij,...j->...", direction, values.h, direction)
    return values.psi - n_dot_xi - 0.5 * n_h_n


def _sample_delta_velocity(
    t_grid, delta_velocity, geometry: LinkGeometry
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(delta_velocity, TestMassPerturbation):
        support_time = np.asarray(_to_numpy(delta_velocity.t), dtype=float)
        values = np.asarray(_to_numpy(delta_velocity.delta_velocity_m_s), dtype=float)
        if support_time.ndim != 1 or len(support_time) < 2:
            raise ValueError("TestMassPerturbation.t must contain at least two samples")
        if np.any(np.diff(support_time) <= 0.0):
            raise ValueError("TestMassPerturbation.t must be strictly increasing")
        if values.shape != (len(support_time), 3, 3):
            raise ValueError(
                "TestMassPerturbation.delta_velocity_m_s must have shape "
                "(len(t), 3, 3)"
            )
        query_min = float(np.min(geometry.t_emission_s))
        query_max = float(np.max(geometry.t_reception_s))
        tolerance = (
            32.0
            * np.finfo(float).eps
            * max(
                1.0,
                abs(query_min),
                abs(query_max),
                abs(support_time[0]),
                abs(support_time[-1]),
            )
        )
        if (
            query_min < support_time[0] - tolerance
            or query_max > support_time[-1] + tolerance
        ):
            raise ValueError(
                "TestMassPerturbation time support must cover all emission and reception events"
            )
        receivers, emitters = _spacecraft_indices(geometry.links)
        delta_receiver = np.empty(geometry.direction.shape, dtype=float)
        delta_emitter = np.empty_like(delta_receiver)
        for link_index, (receiver, emitter) in enumerate(zip(receivers, emitters)):
            delta_receiver[link_index] = _sample_series(
                support_time,
                values[:, receiver, :],
                geometry.t_reception_s,
            )
            delta_emitter[link_index] = _sample_series(
                support_time,
                values[:, emitter, :],
                geometry.t_emission_s[link_index],
            )
        return delta_receiver, delta_emitter

    values = np.asarray(_to_numpy(delta_velocity), dtype=float)
    if values.shape != (len(t_grid), 3, 3):
        raise ValueError("delta_velocity_m_s must have shape (len(t), 3, 3)")
    receivers, emitters = _spacecraft_indices(geometry.links)
    delta_receiver = np.moveaxis(values[:, receivers, :], 0, 1)
    delta_emitter = np.empty_like(delta_receiver)
    for link_index, spacecraft in enumerate(emitters):
        delta_emitter[link_index] = _sample_series(
            np.asarray(t_grid, dtype=float),
            values[:, spacecraft, :],
            geometry.t_emission_s[link_index],
        )
    return delta_receiver, delta_emitter


class WeakFieldLinkResponse:
    """Vectorized finite-arm response on a supplied background orbit."""

    def __init__(
        self,
        *,
        orbits,
        quadrature_order: int = 16,
        chunk_size: int = 32768,
        force_backend: str | None = None,
    ) -> None:
        if int(quadrature_order) < 2:
            raise ValueError("quadrature_order must be at least 2")
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        self.orbits = orbits
        self.quadrature_order = int(quadrature_order)
        self.chunk_size = int(chunk_size)
        self.force_backend = force_backend
        self.backend = _backend_from_name(force_backend)
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        self._nodes = 0.5 * (nodes + 1.0)
        self._weights = 0.5 * weights

    def compute(
        self,
        t_reception_s,
        field: WeakMetricField,
        *,
        delta_velocity_m_s=None,
    ) -> LinkSignalResult:
        """Compute six leading one-way residuals on the reception-time grid."""

        if not isinstance(field, WeakMetricField):
            raise TypeError("field must implement metric() and time_derivative()")
        geometry = build_link_geometry(self.orbits, t_reception_s)
        xp = self.backend.xp
        nlinks = len(geometry.links)
        nt = len(geometry.t_reception_s)
        analytic_direct = getattr(field, "direct_link_signal", None)
        if callable(analytic_direct):
            direct = analytic_direct(geometry, backend=self.backend)
            if direct.shape != (nlinks, nt):
                raise ValueError(
                    "field direct_link_signal() must return shape "
                    f"{(nlinks, nt)}; got {direct.shape}"
                )
        else:
            direct = xp.zeros((nlinks, nt), dtype=xp.float64)
        worldline = xp.zeros_like(direct)
        nodes = xp.asarray(self._nodes)
        weights = xp.asarray(self._weights)

        if delta_velocity_m_s is not None:
            delta_v_r, delta_v_e = _sample_delta_velocity(
                geometry.t_reception_s, delta_velocity_m_s, geometry
            )
        else:
            delta_v_r = delta_v_e = None

        if not callable(analytic_direct) or delta_v_r is not None:
            for start in range(0, nt, self.chunk_size):
                stop = min(start + self.chunk_size, nt)
                sl = slice(start, stop)
                direction = xp.asarray(geometry.direction[:, sl, :])

                if not callable(analytic_direct):
                    t_r = xp.asarray(geometry.t_reception_s[sl])[xp.newaxis, :]
                    t_e = xp.asarray(geometry.t_emission_s[:, sl])
                    light_time = xp.asarray(geometry.light_time_s[:, sl])
                    x_r = xp.asarray(geometry.x_reception_m[:, sl, :])
                    x_e = xp.asarray(geometry.x_emission_m[:, sl, :])
                    metric_e = field.metric(t_e, x_e)
                    metric_r = field.metric(xp.broadcast_to(t_r, t_e.shape), x_r)
                    _validate_metric_values(metric_e, t_e.shape)
                    _validate_metric_values(metric_r, t_e.shape)

                    z = nodes[xp.newaxis, xp.newaxis, :]
                    t_path = t_e[..., xp.newaxis] + light_time[..., xp.newaxis] * z
                    x_path = (
                        x_e[..., xp.newaxis, :]
                        + (x_r - x_e)[..., xp.newaxis, :] * z[..., xp.newaxis]
                    )
                    derivative = field.time_derivative(t_path, x_path)
                    _validate_metric_values(derivative, t_path.shape)
                    path_direction = direction[..., xp.newaxis, :]
                    p_dot = _propagation_kernel(derivative, path_direction, xp)
                    propagation = light_time * xp.sum(
                        p_dot * weights[xp.newaxis, xp.newaxis, :], axis=-1
                    )
                    direct[:, sl] = metric_e.psi - metric_r.psi + propagation

                if delta_v_r is not None and delta_v_e is not None:
                    dv_r = xp.asarray(delta_v_r[:, sl, :])
                    dv_e = xp.asarray(delta_v_e[:, sl, :])
                    worldline[:, sl] = (
                        -xp.einsum("...i,...i->...", direction, dv_r - dv_e) / C_SI
                    )

        total = direct + worldline
        return LinkSignalResult(
            t=geometry.t_reception_s.copy(),
            links=geometry.links,
            direct=direct,
            worldline=worldline,
            total=total,
            geometry=geometry,
            metadata={
                "backend": self.backend.name,
                "force_backend": self.force_backend,
                "quadrature_order": self.quadrature_order,
                "chunk_size": self.chunk_size,
                "response_order": (
                    "linear metric on the fixed photon chord; optional leading "
                    "endpoint velocity"
                ),
                "direct_evaluation": (
                    "analytic" if callable(analytic_direct) else "quadrature"
                ),
                "worldline_velocity_included": delta_velocity_m_s is not None,
                "worldline_velocity_source": (
                    "TestMassPerturbation"
                    if isinstance(delta_velocity_m_s, TestMassPerturbation)
                    else "aligned_array" if delta_velocity_m_s is not None else None
                ),
                "link_order": list(geometry.links),
                "max_null_mismatch": geometry.max_null_mismatch,
                "max_emission_extrapolation_s": geometry.max_emission_extrapolation_s,
            },
        )

    def compute_monochromatic(
        self,
        reference_time_s: float,
        field: MonochromaticWeakMetricField,
    ) -> MonochromaticLinkResult:
        """Return frozen-geometry complex link amplitudes for one mode."""

        if not isinstance(field, MonochromaticWeakMetricField):
            raise TypeError(
                "field must provide angular_frequency and complex_amplitude()"
            )
        t_ref = float(reference_time_s)
        geometry = build_link_geometry(self.orbits, np.asarray([t_ref], dtype=float))
        xp = self.backend.xp
        omega = float(field.angular_frequency)
        phasor_sign = int(field.phasor_sign)
        if phasor_sign not in {-1, 1}:
            raise ValueError("field.phasor_sign must be -1 or +1")
        direction = xp.asarray(geometry.direction[:, 0, :])
        x_e = xp.asarray(geometry.x_emission_m[:, 0, :])
        x_r = xp.asarray(geometry.x_reception_m[:, 0, :])
        light_time = xp.asarray(geometry.light_time_s[:, 0])
        nodes = xp.asarray(self._nodes)
        weights = xp.asarray(self._weights)

        metric_e = field.complex_amplitude(x_e)
        metric_r = field.complex_amplitude(x_r)
        _validate_metric_values(metric_e, (len(geometry.links),))
        _validate_metric_values(metric_r, (len(geometry.links),))

        z = nodes[xp.newaxis, :]
        x_path = (
            x_e[:, xp.newaxis, :] + (x_r - x_e)[:, xp.newaxis, :] * z[..., xp.newaxis]
        )
        metric_path = field.complex_amplitude(x_path)
        _validate_metric_values(metric_path, x_path.shape[:-1])
        p_amp = _propagation_kernel(metric_path, direction[:, xp.newaxis, :], xp)
        retarded_phase = xp.exp(-1j * phasor_sign * omega * light_time)
        path_phase = xp.exp(
            -1j * phasor_sign * omega * light_time[:, xp.newaxis] * (1.0 - z)
        )
        propagation = (
            1j
            * phasor_sign
            * omega
            * light_time
            * xp.sum(p_amp * path_phase * weights[xp.newaxis, :], axis=-1)
        )
        amplitude = retarded_phase * metric_e.psi - metric_r.psi + propagation
        return MonochromaticLinkResult(
            angular_frequency=omega,
            reference_time_s=t_ref,
            links=geometry.links,
            direct_amplitude=amplitude,
            geometry=geometry,
            metadata={
                "backend": self.backend.name,
                "force_backend": self.force_backend,
                "quadrature_order": self.quadrature_order,
                "phasor_sign": phasor_sign,
                "response_order": "linear metric, frozen geometry",
                "worldline_velocity_included": False,
                "link_order": list(geometry.links),
            },
        )


def _xp_for(value):
    module = type(value).__module__.split(".", maxsplit=1)[0]
    if module == "cupy":
        import cupy as cp

        return cp
    return np


@dataclass(frozen=True, kw_only=True)
class ConstantVelocityPointMass:
    """Linear field of a constant-velocity point mass, exact in source speed.

    The position is the source position at ``reference_time_s``.  Source speed
    is supplied once, in SI units, so there is no ambiguity between velocity
    and dimensionless beta.  No finite-size softening is applied.
    """

    rest_mass_kg: float
    position_at_reference_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    reference_time_s: float = 0.0

    def __post_init__(self) -> None:
        mass = float(self.rest_mass_kg)
        position = np.asarray(self.position_at_reference_m, dtype=float)
        velocity = np.asarray(self.velocity_m_s, dtype=float)
        reference_time = float(self.reference_time_s)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("rest_mass_kg must be finite and positive")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_at_reference_m must contain three finite values")
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_m_s must contain three finite values")
        if not np.isfinite(reference_time):
            raise ValueError("reference_time_s must be finite")
        beta_squared = float(np.dot(velocity, velocity) / C_SI**2)
        if beta_squared >= 1.0:
            raise ValueError(
                "velocity_m_s must have magnitude strictly below the speed of light"
            )
        object.__setattr__(self, "rest_mass_kg", mass)
        object.__setattr__(
            self, "position_at_reference_m", tuple(float(value) for value in position)
        )
        object.__setattr__(
            self, "velocity_m_s", tuple(float(value) for value in velocity)
        )
        object.__setattr__(self, "reference_time_s", reference_time)

    @property
    def beta_vector(self) -> np.ndarray:
        """Dimensionless source velocity vector ``velocity_m_s / c``."""

        return np.asarray(self.velocity_m_s, dtype=float) / C_SI

    @property
    def beta(self) -> float:
        """Magnitude of the dimensionless source velocity."""

        return float(np.linalg.norm(self.beta_vector))

    @property
    def lorentz_factor(self) -> float:
        """Source Lorentz factor."""

        return float(1.0 / np.sqrt(1.0 - self.beta**2))

    def position(self, t_s):
        """Return the source position at coordinate time ``t_s``."""

        xp = _xp_for(t_s)
        t = xp.asarray(t_s, dtype=xp.float64)
        position = xp.asarray(self.position_at_reference_m, dtype=xp.float64)
        velocity = xp.asarray(self.velocity_m_s, dtype=xp.float64)
        return position + (t - self.reference_time_s)[..., xp.newaxis] * velocity

    def _profile(self, t_s, x_m):
        xp = _xp_for(x_m)
        x = xp.asarray(x_m, dtype=xp.float64)
        if x.shape[-1:] != (3,):
            raise ValueError("x_m must have final dimension 3")
        t = xp.asarray(t_s, dtype=xp.float64)
        if t.ndim == 1 and x.ndim > 2 and t.shape[0] == x.shape[0]:
            t = t.reshape((len(t),) + (1,) * (x.ndim - 2))
        try:
            t = xp.broadcast_to(t, x.shape[:-1])
        except ValueError as exc:
            raise ValueError("t_s must broadcast to x_m.shape[:-1]") from exc
        position = xp.asarray(self.position_at_reference_m, dtype=xp.float64)
        velocity = xp.asarray(self.velocity_m_s, dtype=xp.float64)
        beta = velocity / C_SI
        gamma_squared = self.lorentz_factor**2
        source_position = (
            position + (t - self.reference_time_s)[..., xp.newaxis] * velocity
        )
        separation = x - source_position
        beta_dot_s = xp.einsum("...i,i->...", separation, beta)
        rho_squared = xp.einsum("...i,...i->...", separation, separation)
        rho_squared = rho_squared + gamma_squared * beta_dot_s**2
        if bool(_to_numpy(xp.any(rho_squared <= 0.0))):
            raise ValueError("point-mass metric is singular on the source worldline")
        rho = xp.sqrt(rho_squared)
        potential = G_SI * self.rest_mass_kg / (C_SI**2 * rho)
        return xp, separation, beta, beta_dot_s, rho, potential, gamma_squared

    @staticmethod
    def _metric_components(xp, beta, gamma_squared: float, scale) -> MetricValues:
        eye = xp.eye(3, dtype=xp.float64)
        beta_outer = beta[:, xp.newaxis] * beta[xp.newaxis, :]
        return MetricValues(
            psi=-(2.0 * gamma_squared - 1.0) * scale,
            xi=-4.0 * gamma_squared * scale[..., xp.newaxis] * beta,
            h=2.0
            * scale[..., xp.newaxis, xp.newaxis]
            * (eye + 2.0 * gamma_squared * beta_outer),
        )

    def metric(self, t_s, x_m) -> MetricValues:
        xp, _s, beta, _beta_dot_s, _rho, potential, gamma_squared = self._profile(
            t_s, x_m
        )
        return self._metric_components(xp, beta, gamma_squared, potential)

    def time_derivative(self, t_s, x_m) -> MetricValues:
        xp, _s, beta, beta_dot_s, rho, _potential, gamma_squared = self._profile(
            t_s, x_m
        )
        potential_dot = (
            G_SI * self.rest_mass_kg / C_SI * gamma_squared * beta_dot_s / rho**3
        )
        return self._metric_components(xp, beta, gamma_squared, potential_dot)

    def acceleration(self, t_s, x_m):
        """Return leading test-mass acceleration on unperturbed positions."""

        xp, separation, beta, beta_dot_s, rho, _potential, gamma_squared = (
            self._profile(t_s, x_m)
        )
        bracket = (2.0 * gamma_squared - 1.0) * separation
        bracket = bracket - (
            gamma_squared
            * (2.0 * gamma_squared + 1.0)
            * beta_dot_s[..., xp.newaxis]
            * beta
        )
        return -G_SI * self.rest_mass_kg * bracket / rho[..., xp.newaxis] ** 3

    def integrate_test_mass_motion(
        self,
        t_s,
        background_position_m,
        *,
        initial_delta_velocity_m_s=None,
        initial_delta_position_m=None,
    ) -> TestMassPerturbation:
        """Integrate leading perturbations along sampled background worldlines.

        The initial perturbations are defined at the first entry of ``t_s``;
        they are independent of the source trajectory reference epoch.
        """

        xp = _xp_for(background_position_m)
        t = xp.asarray(t_s, dtype=xp.float64)
        position = xp.asarray(background_position_m, dtype=xp.float64)
        return _integrate_sampled_acceleration(
            t,
            position,
            self.acceleration(t, position),
            initial_delta_velocity_m_s=initial_delta_velocity_m_s,
            initial_delta_position_m=initial_delta_position_m,
        )

    def direct_link_signal(
        self,
        geometry: LinkGeometry,
        *,
        backend: ArrayBackend | None = None,
    ):
        """Evaluate the exact-in-source-speed linear direct link response.

        The closed photon-path integral is used normally.  A vectorized
        Gauss-Legendre fallback handles geometrically degenerate but nonsingular
        chords without applying a finite-size regularization.
        """

        if not isinstance(geometry, LinkGeometry):
            raise TypeError("geometry must be a LinkGeometry")
        selected_backend = backend or select_array_backend(
            prefer_gpu=False, force="cpu"
        )
        xp = selected_backend.xp
        x_e = xp.asarray(geometry.x_emission_m, dtype=xp.float64)
        x_r = xp.asarray(geometry.x_reception_m, dtype=xp.float64)
        t_e = xp.asarray(geometry.t_emission_s, dtype=xp.float64)
        t_r = xp.asarray(geometry.t_reception_s, dtype=xp.float64)[xp.newaxis, :]
        direction = xp.asarray(geometry.direction, dtype=xp.float64)
        light_time = xp.asarray(geometry.light_time_s, dtype=xp.float64)
        length = xp.asarray(geometry.chord_length_m, dtype=xp.float64)
        beta = xp.asarray(self.beta_vector, dtype=xp.float64)
        gamma_squared = self.lorentz_factor**2

        position = xp.asarray(self.position_at_reference_m, dtype=xp.float64)
        velocity = xp.asarray(self.velocity_m_s, dtype=xp.float64)
        z_e = position + (t_e - self.reference_time_s)[..., xp.newaxis] * velocity
        z_r = position + (t_r - self.reference_time_s)[..., xp.newaxis] * velocity
        s_e = x_e - z_e
        s_r = x_r - z_r
        beta_dot_e = xp.einsum("...i,i->...", s_e, beta)
        beta_dot_r = xp.einsum("...i,i->...", s_r, beta)
        rho_e_squared = (
            xp.einsum("...i,...i->...", s_e, s_e) + gamma_squared * beta_dot_e**2
        )
        rho_r_squared = (
            xp.einsum("...i,...i->...", s_r, s_r) + gamma_squared * beta_dot_r**2
        )
        if bool(_to_numpy(xp.any((rho_e_squared <= 0.0) | (rho_r_squared <= 0.0)))):
            raise ValueError("point-mass metric is singular at a link endpoint")
        rho_e = xp.sqrt(rho_e_squared)
        rho_r = xp.sqrt(rho_r_squared)

        path_time_scale = C_SI * light_time / length
        q = direction - path_time_scale[..., xp.newaxis] * beta
        beta_dot_q = xp.einsum("...i,i->...", q, beta)
        a = xp.einsum("...i,...i->...", q, q) + gamma_squared * beta_dot_q**2
        b = (
            xp.einsum("...i,...i->...", s_e, q)
            + gamma_squared * beta_dot_e * beta_dot_q
        )
        c0 = rho_e_squared
        d = beta_dot_e
        e = beta_dot_q
        discriminant = a * c0 - b**2

        closest_length = xp.clip(-b / a, 0.0, length)
        s_closest = s_e + closest_length[..., xp.newaxis] * q
        beta_dot_closest = xp.einsum("...i,i->...", s_closest, beta)
        rho_closest_squared = (
            xp.einsum("...i,...i->...", s_closest, s_closest)
            + gamma_squared * beta_dot_closest**2
        )
        coordinate_scale = xp.maximum(
            xp.maximum(xp.linalg.norm(x_e, axis=-1), xp.linalg.norm(x_r, axis=-1)),
            xp.maximum(xp.linalg.norm(z_e, axis=-1), xp.linalg.norm(z_r, axis=-1)),
        )
        coordinate_scale = xp.maximum(coordinate_scale, length)
        intersection_tolerance_squared = (
            128.0 * np.finfo(float).eps * coordinate_scale
        ) ** 2
        if bool(
            _to_numpy(xp.any(rho_closest_squared <= intersection_tolerance_squared))
        ):
            raise ValueError("photon chord intersects the point-mass worldline")

        scale = xp.maximum(a * c0, b**2)
        regular = (discriminant > (256.0 * np.finfo(float).eps) * scale) & (
            a > 256.0 * np.finfo(float).eps
        )
        safe_discriminant = xp.where(regular, discriminant, 1.0)
        endpoint_difference = 1.0 / rho_r - 1.0 / rho_e
        boundary = (a * length + b) / rho_r - b / rho_e
        integral = (
            -(e / a) * endpoint_difference
            + (d - e * b / a) / safe_discriminant * boundary
        )

        if bool(_to_numpy(xp.any(~regular))):
            nodes, weights = np.polynomial.legendre.leggauss(64)
            nodes = xp.asarray(0.5 * (nodes + 1.0), dtype=xp.float64)
            weights = xp.asarray(0.5 * weights, dtype=xp.float64)
            fallback_s_e = s_e[~regular]
            fallback_q = q[~regular]
            fallback_length = length[~regular]
            ell = fallback_length[:, xp.newaxis] * nodes[xp.newaxis, :]
            separation = (
                fallback_s_e[:, xp.newaxis, :]
                + ell[..., xp.newaxis] * fallback_q[:, xp.newaxis, :]
            )
            beta_dot_s = xp.einsum("...i,i->...", separation, beta)
            rho_squared = (
                xp.einsum("...i,...i->...", separation, separation)
                + gamma_squared * beta_dot_s**2
            )
            fallback_integral = fallback_length * xp.sum(
                beta_dot_s / rho_squared**1.5 * weights[xp.newaxis, :], axis=-1
            )
            integral[~regular] = fallback_integral

        mass_scale = G_SI * self.rest_mass_kg / C_SI**2
        mu = xp.einsum("...i,i->...", direction, beta)
        endpoint = mass_scale * (2.0 * gamma_squared - 1.0) * endpoint_difference
        propagation = (
            -2.0
            * mass_scale
            * gamma_squared**2
            * (1.0 - mu) ** 2
            * path_time_scale
            * integral
        )
        return endpoint + propagation


@dataclass(frozen=True)
class RetardedQuadrupoleMode:
    """Monochromatic STF mass quadrupole with all radial field zones."""

    angular_frequency: float
    quadrupole_amplitude_kg_m2: np.ndarray
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phasor_sign: int = -1

    def __post_init__(self) -> None:
        omega = float(self.angular_frequency)
        quadrupole = np.asarray(self.quadrupole_amplitude_kg_m2, dtype=np.complex128)
        origin = np.asarray(self.origin_m, dtype=float)
        if not np.isfinite(omega) or omega <= 0.0:
            raise ValueError("angular_frequency must be positive")
        if int(self.phasor_sign) != -1:
            raise ValueError(
                "RetardedQuadrupoleMode uses the documented exp(-i omega t) convention"
            )
        if quadrupole.shape != (3, 3):
            raise ValueError("quadrupole_amplitude_kg_m2 must have shape (3, 3)")
        if not np.all(np.isfinite(quadrupole)):
            raise ValueError("quadrupole_amplitude_kg_m2 must contain finite values")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_m must contain three finite values")
        scale = max(1.0, float(np.max(np.abs(quadrupole))))
        if not np.allclose(
            quadrupole, quadrupole.T, rtol=1.0e-12, atol=1.0e-12 * scale
        ):
            raise ValueError("quadrupole tensor must be symmetric")
        if not np.isclose(
            np.trace(quadrupole), 0.0, rtol=1.0e-12, atol=1.0e-12 * scale
        ):
            raise ValueError("quadrupole tensor must be trace free")
        object.__setattr__(self, "angular_frequency", omega)
        object.__setattr__(self, "quadrupole_amplitude_kg_m2", quadrupole)
        object.__setattr__(
            self, "origin_m", tuple(float(value) for value in origin)
        )

    def complex_amplitude(self, x_m) -> MetricValues:
        xp = _xp_for(x_m)
        x = xp.asarray(x_m, dtype=xp.float64)
        if x.shape[-1:] != (3,):
            raise ValueError("x_m must have final dimension 3")
        origin = xp.asarray(self.origin_m, dtype=xp.float64)
        displacement = x - origin
        radius = xp.linalg.norm(displacement, axis=-1)
        if bool(_to_numpy(xp.any(radius <= 0.0))):
            raise ValueError("quadrupole metric is singular at its source origin")
        radial = displacement / radius[..., xp.newaxis]
        omega = self.angular_frequency
        quadrupole = xp.asarray(self.quadrupole_amplitude_kg_m2)
        i_amp = (
            quadrupole * xp.exp(1j * omega * radius / C_SI)[..., xp.newaxis, xp.newaxis]
        )
        i_dot = -1j * omega * i_amp
        i_ddot = -(omega**2) * i_amp

        radial_contraction = xp.einsum(
            "...i,...j,...ij->...",
            radial,
            radial,
            3.0 * i_amp / radius[..., xp.newaxis, xp.newaxis] ** 3
            + 3.0 * i_dot / (C_SI * radius[..., xp.newaxis, xp.newaxis] ** 2)
            + i_ddot / (C_SI**2 * radius[..., xp.newaxis, xp.newaxis]),
        )
        h00 = G_SI / C_SI**2 * radial_contraction
        h0 = (
            -2.0
            * G_SI
            / C_SI**3
            * xp.einsum(
                "...j,...ij->...i",
                radial,
                i_dot / radius[..., xp.newaxis, xp.newaxis] ** 2
                + i_ddot / (C_SI * radius[..., xp.newaxis, xp.newaxis]),
            )
        )
        eye = xp.eye(3, dtype=xp.float64)
        hij = h00[..., xp.newaxis, xp.newaxis] * eye + (
            2.0 * G_SI / (C_SI**4 * radius[..., xp.newaxis, xp.newaxis]) * i_ddot
        )
        return MetricValues(psi=-0.5 * h00, xi=h0, h=hij)

    def metric(self, t_s, x_m) -> MetricValues:
        xp = _xp_for(x_m)
        t = xp.asarray(t_s, dtype=xp.float64)
        amplitude = self.complex_amplitude(x_m)
        phase = xp.exp(-1j * self.angular_frequency * t)
        return MetricValues(
            psi=xp.real(amplitude.psi * phase),
            xi=xp.real(amplitude.xi * phase[..., xp.newaxis]),
            h=xp.real(amplitude.h * phase[..., xp.newaxis, xp.newaxis]),
        )

    def time_derivative(self, t_s, x_m) -> MetricValues:
        xp = _xp_for(x_m)
        t = xp.asarray(t_s, dtype=xp.float64)
        amplitude = self.complex_amplitude(x_m)
        factor = -1j * self.angular_frequency * xp.exp(-1j * self.angular_frequency * t)
        return MetricValues(
            psi=xp.real(amplitude.psi * factor),
            xi=xp.real(amplitude.xi * factor[..., xp.newaxis]),
            h=xp.real(amplitude.h * factor[..., xp.newaxis, xp.newaxis]),
        )

    def acceleration_complex_amplitude(self, x_m):
        """Return the leading nonrelativistic test-mass acceleration phasor."""

        xp = _xp_for(x_m)
        x = xp.asarray(x_m, dtype=xp.float64)
        if x.shape[-1:] != (3,):
            raise ValueError("x_m must have final dimension 3")
        origin = xp.asarray(self.origin_m, dtype=xp.float64)
        displacement = x - origin
        radius = xp.linalg.norm(displacement, axis=-1)
        if bool(_to_numpy(xp.any(radius <= 0.0))):
            raise ValueError("quadrupole metric is singular at its source origin")

        radial = displacement / radius[..., xp.newaxis]
        omega = self.angular_frequency
        wave_number = omega / C_SI
        quadrupole = xp.asarray(self.quadrupole_amplitude_kg_m2)
        qn = xp.einsum("ij,...j->...i", quadrupole, radial)
        nqn = xp.einsum("...i,ij,...j->...", radial, quadrupole, radial)
        retarded_phase = xp.exp(1j * wave_number * radius)

        radial_kernel = (
            3.0 / radius**3
            - 3j * wave_number / radius**2
            - wave_number**2 / radius
        )
        radial_kernel_derivative = (
            -9.0 / radius**4
            + 6j * wave_number / radius**3
            + wave_number**2 / radius**2
        )
        grad_h00 = (
            G_SI
            / C_SI**2
            * retarded_phase[..., xp.newaxis]
            * (
                radial
                * nqn[..., xp.newaxis]
                * (
                    1j * wave_number * radial_kernel
                    + radial_kernel_derivative
                )[..., xp.newaxis]
                + 2.0
                * (radial_kernel / radius)[..., xp.newaxis]
                * (qn - nqn[..., xp.newaxis] * radial)
            )
        )
        xi_amplitude = (
            -2.0
            * G_SI
            / C_SI**3
            * retarded_phase[..., xp.newaxis]
            * qn
            * xp.asarray(
                -1j * omega / radius**2
                - omega**2 / (C_SI * radius)
            )[..., xp.newaxis]
        )
        grad_psi = -0.5 * grad_h00
        return -C_SI**2 * grad_psi + 1j * C_SI * omega * xi_amplitude

    def steady_state_test_mass_motion(
        self,
        t_s,
        background_position_m,
    ) -> TestMassPerturbation:
        """Return the leading oscillatory motion on background worldlines.

        The result uses the monochromatic forced-motion solution and neglects
        derivatives of the slowly varying background geometry.  It is intended
        for source frequencies well above the background orbital frequency.
        """

        xp = _xp_for(background_position_m)
        t = xp.asarray(t_s, dtype=xp.float64)
        position = xp.asarray(background_position_m, dtype=xp.float64)
        if t.ndim != 1 or len(t) < 2:
            raise ValueError(
                "t_s must be a one-dimensional array with at least two samples"
            )
        if position.shape[0] != len(t) or position.shape[-1:] != (3,):
            raise ValueError("background_position_m must have shape (len(t_s), ..., 3)")
        if bool(_to_numpy(xp.any(xp.diff(t) <= 0.0))):
            raise ValueError("t_s must be strictly increasing")

        acceleration_amplitude = self.acceleration_complex_amplitude(position)
        phase_shape = (len(t),) + (1,) * (position.ndim - 2)
        phase = xp.exp(-1j * self.angular_frequency * t.reshape(phase_shape))
        acceleration = xp.real(acceleration_amplitude * phase[..., xp.newaxis])
        delta_velocity = xp.real(
            1j
            * acceleration_amplitude
            / self.angular_frequency
            * phase[..., xp.newaxis]
        )
        delta_position = xp.real(
            -acceleration_amplitude
            / self.angular_frequency**2
            * phase[..., xp.newaxis]
        )
        return TestMassPerturbation(
            t=t,
            acceleration_m_s2=acceleration,
            delta_velocity_m_s=delta_velocity,
            delta_position_m=delta_position,
        )


@dataclass(frozen=True)
class SmoothVaidyaMassLoss:
    """Smooth outgoing mass-loss perturbation relative to the pre-loss field."""

    delta_mass_kg: float
    transition_time_s: float
    center_retarded_time_s: float = 0.0
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        delta_mass = float(self.delta_mass_kg)
        transition_time = float(self.transition_time_s)
        center_time = float(self.center_retarded_time_s)
        origin = np.asarray(self.origin_m, dtype=float)
        if not np.isfinite(delta_mass) or delta_mass <= 0.0:
            raise ValueError("delta_mass_kg must be finite and positive")
        if not np.isfinite(transition_time) or transition_time <= 0.0:
            raise ValueError("transition_time_s must be finite and positive")
        if not np.isfinite(center_time):
            raise ValueError("center_retarded_time_s must be finite")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_m must contain three finite values")
        object.__setattr__(self, "delta_mass_kg", delta_mass)
        object.__setattr__(self, "transition_time_s", transition_time)
        object.__setattr__(self, "center_retarded_time_s", center_time)
        object.__setattr__(
            self, "origin_m", tuple(float(value) for value in origin)
        )

    def _profile(self, t_s, x_m):
        xp = _xp_for(x_m)
        x = xp.asarray(x_m, dtype=xp.float64)
        if x.shape[-1:] != (3,):
            raise ValueError("x_m must have final dimension 3")
        displacement = x - xp.asarray(self.origin_m, dtype=xp.float64)
        radius = xp.linalg.norm(displacement, axis=-1)
        if bool(_to_numpy(xp.any(radius <= 0.0))):
            raise ValueError("Vaidya metric is singular at its source origin")
        radial = displacement / radius[..., xp.newaxis]
        retarded_time = xp.asarray(t_s, dtype=xp.float64) - radius / C_SI
        argument = (retarded_time - float(self.center_retarded_time_s)) / float(
            self.transition_time_s
        )
        tanh_argument = xp.tanh(argument)
        profile = 0.5 * (1.0 + tanh_argument)
        profile_dot = 0.5 * (1.0 - tanh_argument**2) / float(self.transition_time_s)
        amplitude = G_SI * float(self.delta_mass_kg) / (C_SI**2 * radius)
        return xp, radial, radius, amplitude, profile, profile_dot

    @staticmethod
    def _components(xp, radial, scale) -> MetricValues:
        return MetricValues(
            psi=scale,
            xi=2.0 * scale[..., xp.newaxis] * radial,
            h=-2.0
            * scale[..., xp.newaxis, xp.newaxis]
            * radial[..., :, xp.newaxis]
            * radial[..., xp.newaxis, :],
        )

    def metric(self, t_s, x_m) -> MetricValues:
        xp, radial, _radius, amplitude, profile, _profile_dot = self._profile(
            t_s, x_m
        )
        return self._components(xp, radial, amplitude * profile)

    def time_derivative(self, t_s, x_m) -> MetricValues:
        xp, radial, _radius, amplitude, _profile, profile_dot = self._profile(
            t_s, x_m
        )
        return self._components(xp, radial, amplitude * profile_dot)

    def acceleration(self, t_s, x_m):
        """Return the leading nonrelativistic acceleration of a test mass.

        This is the fixed-background result for the perturbation relative to
        the pre-loss field; it includes the outward post-loss acceleration and
        the inward impulse carried by the outgoing null shell.
        """

        xp, radial, radius, _amplitude, profile, profile_dot = self._profile(
            t_s, x_m
        )
        scale = G_SI * float(self.delta_mass_kg) * (
            profile / radius**2 - profile_dot / (C_SI * radius)
        )
        return scale[..., xp.newaxis] * radial

    def integrate_test_mass_motion(
        self,
        t_s,
        background_position_m,
        *,
        initial_delta_velocity_m_s=None,
        initial_delta_position_m=None,
    ) -> TestMassPerturbation:
        """Integrate leading perturbations along sampled background worldlines."""

        xp = _xp_for(background_position_m)
        t = xp.asarray(t_s, dtype=xp.float64)
        position = xp.asarray(background_position_m, dtype=xp.float64)
        return _integrate_sampled_acceleration(
            t,
            position,
            self.acceleration(t, position),
            initial_delta_velocity_m_s=initial_delta_velocity_m_s,
            initial_delta_position_m=initial_delta_position_m,
        )


__all__ = [
    "G_SI",
    "LinkGeometry",
    "LinkSignalResult",
    "MetricValues",
    "MonochromaticLinkResult",
    "MonochromaticWeakMetricField",
    "RetardedQuadrupoleMode",
    "SmoothVaidyaMassLoss",
    "TestMassPerturbation",
    "ConstantVelocityPointMass",
    "WeakFieldLinkResponse",
    "WeakMetricField",
    "build_link_geometry",
]
