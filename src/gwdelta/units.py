"""Physical-unit helpers for feeding geometric-unit waveforms to TDI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.98847e30
PC_SI = 3.0856775814913673e16

DISTANCE_UNITS_SI = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "pc": PC_SI,
    "kpc": 1.0e3 * PC_SI,
    "mpc": 1.0e6 * PC_SI,
    "gpc": 1.0e9 * PC_SI,
}


@dataclass(frozen=True)
class WaveformSamples:
    """Lightweight waveform container used by unit-conversion helpers."""

    t: Any
    h_plus: Any
    h_cross: Any | None
    backend: Any
    metadata: dict[str, Any]

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return (
            np.asarray(self.t),
            np.asarray(self.h_plus),
            None if self.h_cross is None else np.asarray(self.h_cross),
        )


def distance_to_meters(value: float, unit: str = "m") -> float:
    """Return a distance in meters."""

    key = unit.lower()
    if key not in DISTANCE_UNITS_SI:
        raise ValueError(f"unsupported distance unit: {unit}")
    distance = float(value) * DISTANCE_UNITS_SI[key]
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    return distance


@dataclass(frozen=True)
class PhysicalScale:
    """Map code units with total mass ``code_total_mass`` to SI units.

    If the waveform code uses ``total_mass=1``, one code-time unit is
    ``G M_phys / c^3`` and one code-length unit is ``G M_phys / c^2``.
    For a different dimensionless code mass, both units are divided by
    ``code_total_mass``.
    """

    total_mass_solar: float
    distance_m: float
    code_total_mass: float = 1.0

    def __post_init__(self) -> None:
        if self.total_mass_solar <= 0.0:
            raise ValueError("total_mass_solar must be positive")
        if self.distance_m <= 0.0:
            raise ValueError("distance_m must be positive")
        if self.code_total_mass <= 0.0:
            raise ValueError("code_total_mass must be positive")

    @property
    def total_mass_kg(self) -> float:
        return self.total_mass_solar * M_SUN_SI

    @property
    def time_unit_s(self) -> float:
        return G_SI * self.total_mass_kg / C_SI**3 / self.code_total_mass

    @property
    def length_unit_m(self) -> float:
        return G_SI * self.total_mass_kg / C_SI**2 / self.code_total_mass

    @property
    def strain_scale(self) -> float:
        return self.length_unit_m / self.distance_m

    def time_to_seconds(self, t_code: Any) -> np.ndarray:
        return np.asarray(t_code, dtype=float) * self.time_unit_s

    def seconds_to_code_time(self, t_seconds: Any) -> np.ndarray:
        return np.asarray(t_seconds, dtype=float) / self.time_unit_s

    def strain_to_physical(self, h_code: Any) -> np.ndarray:
        return np.asarray(h_code, dtype=float) * self.strain_scale

    def samples_to_physical(self, samples: WaveformSamples) -> WaveformSamples:
        """Return a NumPy ``WaveformSamples`` object scaled for TDI."""

        t, hp, hc = samples.as_numpy()
        return WaveformSamples(
            t=self.time_to_seconds(t),
            h_plus=self.strain_to_physical(hp),
            h_cross=None if hc is None else self.strain_to_physical(hc),
            backend=samples.backend,
            metadata={
                **samples.metadata,
                "time_unit_s": self.time_unit_s,
                "length_unit_m": self.length_unit_m,
                "strain_scale": self.strain_scale,
                "total_mass_solar": self.total_mass_solar,
                "distance_m": self.distance_m,
                "code_total_mass": self.code_total_mass,
            },
        )


def make_physical_scale(
    *,
    total_mass_solar: float,
    distance: float,
    distance_unit: str = "Mpc",
    code_total_mass: float = 1.0,
) -> PhysicalScale:
    """Construct a ``PhysicalScale`` from common astronomy units."""

    return PhysicalScale(
        total_mass_solar=float(total_mass_solar),
        distance_m=distance_to_meters(distance, distance_unit),
        code_total_mass=float(code_total_mass),
    )
