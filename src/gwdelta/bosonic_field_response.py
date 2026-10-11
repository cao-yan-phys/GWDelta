from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .array_backend import infer_backend_from_array, select_array_backend
from .cuda_runtime import ensure_cuda_dll_directories
from .fd_response import C_SI, DEFAULT_LINKS, receivers_emitters
from .weak_field import LinkSignalResult, build_link_geometry


def _numpy(value):
    return infer_backend_from_array(value).asnumpy(value)


def _parameters(frequencies, amplitudes, direction, mass, reference_time, reference_position, vector=False):
    frequencies = np.atleast_1d(np.asarray(_numpy(frequencies), dtype=float)).copy()
    amplitudes = np.asarray(_numpy(amplitudes), dtype=complex)
    shape = (len(frequencies), 3) if vector else (len(frequencies),)
    if amplitudes.shape == ((3,) if vector else ()) and len(frequencies) == 1:
        amplitudes = amplitudes.reshape(shape)
    if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_hz must be a nonempty finite vector")
    if np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies_hz must be positive and strictly increasing")
    if not np.isfinite(mass) or mass < 0 or np.any(frequencies < mass):
        raise ValueError("frequencies must lie in the propagating domain of mass_frequency_hz")
    if amplitudes.shape != shape or not np.all(np.isfinite(amplitudes)):
        raise ValueError(f"amplitudes must be finite with shape {shape}")
    direction = np.asarray(direction, dtype=float)
    position = np.asarray(reference_position, dtype=float).copy()
    norm = np.linalg.norm(direction)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or not np.isfinite(norm) or norm == 0:
        raise ValueError("direction must be a nonzero finite three-vector")
    if position.shape != (3,) or not np.all(np.isfinite(position)) or not np.isfinite(reference_time):
        raise ValueError("reference coordinates must be finite")
    direction = direction / norm
    amplitudes = amplitudes.copy()
    for value in (frequencies, amplitudes, direction, position):
        value.setflags(write=False)
    return frequencies, amplitudes, direction, float(mass), float(reference_time), position


def _spectrum(frequencies, spectrum, mass, vector=False):
    frequencies = np.asarray(_numpy(frequencies), dtype=float)
    spectrum = np.asarray(_numpy(spectrum), dtype=complex)
    if frequencies.ndim != 1 or len(frequencies) < 2 or not np.all(np.isfinite(frequencies)):
        raise ValueError("a continuous spectrum requires at least two finite frequency samples")
    if np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("spectrum frequencies must be positive and strictly increasing")
    shape = (len(frequencies), 3) if vector else frequencies.shape
    if spectrum.shape != shape or not np.all(np.isfinite(spectrum)):
        raise ValueError(f"spectrum must be finite with shape {shape}")
    if not np.isfinite(mass) or mass < 0:
        raise ValueError("mass_frequency_hz must be finite and nonnegative")
    below = frequencies < mass
    if np.any(spectrum[below] != 0):
        raise ValueError("nonzero subthreshold spectra are not propagating plane waves")
    keep = ~below
    if not np.any(keep):
        raise ValueError("spectrum has no propagating frequency samples")
    widths = np.diff(frequencies)
    weights = np.concatenate(([widths[0]/2], (widths[:-1]+widths[1:])/2, [widths[-1]/2]))
    amplitudes = 2*spectrum*weights.reshape((-1, 1) if vector else (-1,))
    return frequencies[keep], amplitudes[keep]


@dataclass(frozen=True)
class ScalarPlaneWave:
    frequencies_hz: object
    amplitudes: object
    direction: object
    mass_frequency_hz: float = 0.0
    reference_time_s: float = 0.0
    reference_position_m: object = (0.0, 0.0, 0.0)

    def __post_init__(self):
        values = _parameters(self.frequencies_hz, self.amplitudes, self.direction,
                             self.mass_frequency_hz, self.reference_time_s, self.reference_position_m)
        for name, value in zip(self.__dataclass_fields__, values):
            object.__setattr__(self, name, value)

    @classmethod
    def from_spectrum(cls, frequencies_hz, spectrum, direction, *, mass_frequency_hz=0.0, **kwargs):
        frequencies, amplitudes = _spectrum(frequencies_hz, spectrum, mass_frequency_hz)
        return cls(frequencies, amplitudes, direction, mass_frequency_hz, **kwargs)

    def velocity_coefficients(self):
        vg = np.sqrt((1-self.mass_frequency_hz/self.frequencies_hz)*(1+self.mass_frequency_hz/self.frequencies_hz))
        return -vg[:, None]*self.amplitudes[:, None]*self.direction


@dataclass(frozen=True)
class VectorPlaneWave:
    frequencies_hz: object
    amplitudes: object
    direction: object
    mass_frequency_hz: float = 0.0
    reference_time_s: float = 0.0
    reference_position_m: object = (0.0, 0.0, 0.0)
    temporal_amplitudes: object = None

    def __post_init__(self):
        values = _parameters(self.frequencies_hz, self.amplitudes, self.direction,
                             self.mass_frequency_hz, self.reference_time_s, self.reference_position_m, True)
        for name, value in zip(tuple(self.__dataclass_fields__)[:6], values):
            object.__setattr__(self, name, value)
        vg = np.sqrt((1-self.mass_frequency_hz/self.frequencies_hz)*(1+self.mass_frequency_hz/self.frequencies_hz))
        expected = vg*(self.amplitudes@self.direction)
        if self.temporal_amplitudes is not None:
            supplied = np.atleast_1d(np.asarray(_numpy(self.temporal_amplitudes), dtype=complex))
            scale = np.maximum(np.linalg.norm(self.amplitudes, axis=1), np.finfo(float).tiny)
            if supplied.shape != expected.shape or not np.all(np.isfinite(supplied)):
                raise ValueError("temporal_amplitudes must match frequencies_hz")
            if np.any(np.abs(supplied-expected) > 1e-12*scale):
                raise ValueError("temporal_amplitudes violate the Lorenz condition")
        expected.setflags(write=False)
        object.__setattr__(self, "temporal_amplitudes", expected)

    @classmethod
    def from_spectrum(cls, frequencies_hz, spectrum, direction, *, mass_frequency_hz=0.0, **kwargs):
        frequencies, amplitudes = _spectrum(frequencies_hz, spectrum, mass_frequency_hz, True)
        return cls(frequencies, amplitudes, direction, mass_frequency_hz, **kwargs)

    def velocity_coefficients(self):
        vg = np.sqrt((1-self.mass_frequency_hz/self.frequencies_hz)*(1+self.mass_frequency_hz/self.frequencies_hz))
        longitudinal = self.amplitudes@self.direction
        transverse = self.amplitudes-longitudinal[:, None]*self.direction
        return -transverse-(self.mass_frequency_hz/self.frequencies_hz)[:, None]**2*longitudinal[:, None]*self.direction


def static_link_amplitudes(field, positions_m, *, links=DEFAULT_LINKS):
    if not isinstance(field, (ScalarPlaneWave, VectorPlaneWave)):
        raise TypeError("field must be ScalarPlaneWave or VectorPlaneWave")
    positions = np.asarray(_numpy(positions_m), dtype=float)
    links = tuple(int(link) for link in links)
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)) or set(links) != set(DEFAULT_LINKS) or len(links) != 6:
        raise ValueError("positions and six directed links must define a finite triangle")
    receivers, emitters = receivers_emitters(links)
    chord = positions[receivers]-positions[emitters]
    length = np.linalg.norm(chord, axis=1)
    if np.any(length == 0):
        raise ValueError("spacecraft must have distinct positions")
    n = chord/length[:, None]
    omega = 2*np.pi*field.frequencies_hz
    vg = np.sqrt((1-field.mass_frequency_hz/field.frequencies_hz)*(1+field.mass_frequency_hz/field.frequencies_hz))
    phase = np.exp(1j*omega*(field.reference_time_s+vg*((positions[receivers]-field.reference_position_m)@field.direction)[:, None]/C_SI))
    difference = -np.expm1(1j*omega*length[:, None]/C_SI*(1-(n@field.direction)[:, None]*vg))
    return -(n@field.velocity_coefficients().T)*phase*difference


class BosonicFieldLinkResponse:
    def __init__(self, orbits, *, force_backend="cpu", chunk_size=2048, frequency_chunk_size=64):
        if force_backend not in ("cpu", "cupy", "cuda12x"):
            raise ValueError("force_backend must be cpu, cupy, or cuda12x")
        if chunk_size < 1 or frequency_chunk_size < 1 or int(chunk_size) != chunk_size or int(frequency_chunk_size) != frequency_chunk_size:
            raise ValueError("chunk sizes must be positive integers")
        if force_backend != "cpu":
            ensure_cuda_dll_directories()
        self.backend = select_array_backend(force="cpu" if force_backend == "cpu" else "cupy")
        if force_backend != "cpu" and not self.backend.is_gpu:
            raise RuntimeError("requested GPU backend is unavailable")
        self.orbits = orbits
        self.chunk_size = int(chunk_size)
        self.frequency_chunk_size = int(frequency_chunk_size)

    def _increments(self, field, time, positions, duration, displacement):
        xp = self.backend.xp
        t = xp.asarray(time)
        p = xp.asarray(positions)
        dt = xp.asarray(duration)
        dx = xp.asarray(displacement)
        result = xp.zeros(p.shape, dtype=xp.float64)
        coefficients = field.velocity_coefficients()
        for start in range(0, len(field.frequencies_hz), self.frequency_chunk_size):
            sl = slice(start, start+self.frequency_chunk_size)
            omega = xp.asarray(2*np.pi*field.frequencies_hz[sl])
            ratio = field.mass_frequency_hz/field.frequencies_hz[sl]
            vg = xp.asarray(np.sqrt((1-ratio)*(1+ratio)))
            k = xp.asarray(field.direction)
            phase = (t-field.reference_time_s)[:, None]-((p-xp.asarray(field.reference_position_m))@k)[:, None]*vg/C_SI
            advance = omega*(dt[:, None]-(dx@k)[:, None]*vg/C_SI)
            factor = xp.exp(-1j*(omega*phase+advance/2))*xp.sinc(advance/(2*np.pi))*(-1j*omega*dt[:, None])
            result += C_SI*xp.real(factor@xp.asarray(coefficients[sl]))
        return result

    def compute(self, t_reception_s, field, *, initial_time_s=None, initial_velocity_m_s=None, integration_step_s=60.0):
        if not isinstance(field, (ScalarPlaneWave, VectorPlaneWave)):
            raise TypeError("field must be ScalarPlaneWave or VectorPlaneWave")
        if not np.isfinite(integration_step_s) or integration_step_s <= 0:
            raise ValueError("integration_step_s must be finite and positive")
        geometry = build_link_geometry(self.orbits, t_reception_s)
        base_t = np.asarray(_numpy(self.orbits.t_base), dtype=float)
        base_x = np.asarray(_numpy(self.orbits.x_base), dtype=float)
        start = float(base_t[0] if initial_time_s is None else initial_time_s)
        end = float(geometry.t_reception_s[-1])
        if not np.isfinite(start) or start < base_t[0] or start > np.min(geometry.t_emission_s) or end <= start:
            raise ValueError("initial time and all retarded events must lie within orbit support")
        initial = np.zeros((3, 3)) if initial_velocity_m_s is None else np.asarray(_numpy(initial_velocity_m_s), dtype=float)
        if initial.shape != (3, 3) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_velocity_m_s must be a finite (3, 3) array")
        count = max(1, int(np.ceil((end-start)/integration_step_s)))
        grid = np.linspace(0., end-start, count+1)
        durations = np.diff(grid)
        spline = CubicSpline(base_t-start, base_x, axis=0, extrapolate=False)
        positions = spline(grid)
        xp = self.backend.xp
        motion = xp.empty((count+1, 3, 3), dtype=xp.float64)
        motion[0] = xp.asarray(initial)
        for sc in range(3):
            accumulated = xp.asarray(initial[sc])
            for lower in range(0, count, self.chunk_size):
                upper = min(count, lower+self.chunk_size)
                inc = self._increments(field, grid[lower:upper]+start, positions[lower:upper, sc],
                                       durations[lower:upper], np.diff(positions[lower:upper+1, sc], axis=0))
                values = xp.cumsum(inc, axis=0)+accumulated
                motion[lower+1:upper+1, sc] = values
                accumulated = values[-1].copy()
        def velocity(query, sc):
            query = np.asarray(query)-start
            index = np.clip(np.searchsorted(grid, query, side="right")-1, 0, count-1)
            duration = query-grid[index]
            position = positions[index, sc]
            fraction = duration/(grid[index+1]-grid[index])
            displacement = (positions[index+1, sc]-position)*fraction[:, None]
            result = motion[xp.asarray(index), sc].copy()
            for lower in range(0, len(query), self.chunk_size):
                sl = slice(lower, lower+self.chunk_size)
                result[sl] += self._increments(field, grid[index[sl]]+start, position[sl], duration[sl], displacement[sl])
            return result
        total = xp.empty((6, len(geometry.t_reception_s)), dtype=xp.float64)
        receivers, emitters = receivers_emitters(geometry.links)
        for link, (receiver, emitter) in enumerate(zip(receivers, emitters)):
            vr = velocity(geometry.t_reception_s, receiver)
            ve = velocity(geometry.t_emission_s[link], emitter)
            total[link] = -xp.sum(xp.asarray(geometry.direction[link])*(vr-ve), axis=-1)/C_SI
        return LinkSignalResult(geometry.t_reception_s, geometry.links, xp.zeros_like(total), total, total,
                                geometry, {"backend": self.backend.name, "input_kind": "bosonic_field",
                                           "field_model": type(field).__name__, "fourier_convention": "forward exp(+2piift); inverse exp(-2piift)",
                                           "initial_time_s": start, "integration_step_s": float(grid[1]-grid[0]),
                                           "integration_intervals": count, "force_order": "velocity-independent",
                                           "trajectory_integration": "piecewise-linear exact mode integrals"})
