"""Frozen-arm unequal-length second-generation TDI noise covariances.

The implementation follows the primitive-noise construction used for the
secondary-noise model in Hartwig et al., Phys. Rev. D 107, 123531 (2023),
arXiv:2303.15929.  It is deliberately restricted to static/frozen delays:
the delay operators commute, while the six one-way light times and the OMS/TM
noise PSDs may still be unequal.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .fd_response import C_SI, DEFAULT_LINKS, SECOND_GEN_X_COMBINATIONS, cyclic_permutation
from .noise import DetectorNoiseModel, one_way_noise_psd


_LINKS = tuple((int(link) // 10, int(link) % 10) for link in DEFAULT_LINKS)
_LINK_INDEX = {link: index for index, link in enumerate(_LINKS)}
_AET_FROM_XYZ = np.asarray(
    [
        (-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)),
        (1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)),
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
    ],
    dtype=float,
)


def _link_from_key(key: Any) -> tuple[int, int]:
    if isinstance(key, Integral):
        value = int(key)
        link = (value // 10, value % 10)
    else:
        try:
            first, second = key
        except (TypeError, ValueError) as exc:
            raise ValueError("link keys must be two-tuples or labels such as 12") from exc
        link = (int(first), int(second))
    if link not in _LINK_INDEX:
        raise ValueError(f"unknown directed link {key!r}; expected the six inter-spacecraft links")
    return link


def _light_time_map(light_times_s: Mapping[Any, float]) -> dict[tuple[int, int], float]:
    if not isinstance(light_times_s, Mapping):
        raise ValueError("light_times_s must map all six directed links to light times in seconds")
    values: dict[tuple[int, int], float] = {}
    for key, value in light_times_s.items():
        link = _link_from_key(key)
        if link in values:
            raise ValueError(f"duplicate light time for link {link[0]}{link[1]}")
        value_float = float(value)
        if not np.isfinite(value_float) or value_float <= 0.0:
            raise ValueError("all light times must be finite and positive")
        values[link] = value_float
    missing = [link for link in _LINKS if link not in values]
    if missing:
        labels = ", ".join(f"{i}{j}" for i, j in missing)
        raise ValueError(f"light_times_s is missing directed links: {labels}")
    return values


def _frequency_array(frequency_hz: float | np.ndarray) -> tuple[np.ndarray, bool]:
    frequency = np.asarray(frequency_hz, dtype=float)
    if frequency.ndim > 1:
        raise ValueError("frequency_hz must be a scalar or one-dimensional array")
    scalar = frequency.ndim == 0
    frequency = np.atleast_1d(frequency)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must be finite and strictly positive")
    return frequency, scalar


def _psd_array(value: Any, frequency_size: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(frequency_size, float(array), dtype=float)
    elif array.shape != (frequency_size,):
        raise ValueError(f"{name} must be scalar or have shape ({frequency_size},)")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def _link_psds(
    values: Any,
    frequency_size: int,
    *,
    name: str,
) -> np.ndarray:
    if not isinstance(values, Mapping):
        common = _psd_array(values, frequency_size, name=name)
        return np.broadcast_to(common, (len(_LINKS), frequency_size)).copy()

    by_link: dict[tuple[int, int], Any] = {}
    for key, value in values.items():
        link = _link_from_key(key)
        if link in by_link:
            raise ValueError(f"duplicate {name} for link {link[0]}{link[1]}")
        by_link[link] = value
    missing = [link for link in _LINKS if link not in by_link]
    if missing:
        labels = ", ".join(f"{i}{j}" for i, j in missing)
        raise ValueError(f"{name} is missing directed links: {labels}")
    return np.stack(
        [_psd_array(by_link[link], frequency_size, name=f"{name}[{link[0]}{link[1]}]") for link in _LINKS]
    )


def frozen_tdi2_light_times_from_positions(positions_m: np.ndarray) -> dict[tuple[int, int], float]:
    """Return reciprocal frozen one-way light times from three SSB positions."""

    positions = np.asarray(positions_m, dtype=float)
    if positions.shape != (3, 3) or np.any(~np.isfinite(positions)):
        raise ValueError("positions_m must be a finite array with shape (3, 3)")
    return {
        (receiver, emitter): float(
            np.linalg.norm(positions[receiver - 1] - positions[emitter - 1]) / C_SI
        )
        for receiver, emitter in _LINKS
    }


def _delay_phases(frequency_hz: np.ndarray, light_times_s: Mapping[tuple[int, int], float]) -> np.ndarray:
    light_times = np.asarray([light_times_s[link] for link in _LINKS], dtype=float)
    return np.exp(-2j * np.pi * frequency_hz[:, np.newaxis] * light_times[np.newaxis, :])


def _cleaned_link_map(delay_phases: np.ndarray) -> np.ndarray:
    """Return the ``eta = B n`` map for OMS/TM primitive-noise ordering."""

    frequency_size = delay_phases.shape[0]
    mapping = np.zeros((frequency_size, 6, 12), dtype=np.complex128)
    for index, (receiver, emitter) in enumerate(_LINKS):
        opposite = _LINK_INDEX[(emitter, receiver)]
        sign = 1.0 if (receiver, emitter) in {(1, 3), (2, 1), (3, 2)} else -1.0
        mapping[:, index, index] = 1.0
        mapping[:, index, 6 + index] = sign
        mapping[:, index, 6 + opposite] = -sign * delay_phases[:, index]
    return mapping


def _xyz_coefficients(
    frequency_hz: np.ndarray,
    light_times_s: Mapping[tuple[int, int], float],
) -> np.ndarray:
    """Return frozen-arm second-generation XYZ link coefficients."""

    coefficients = np.zeros((len(frequency_hz), 3, 6), dtype=np.complex128)
    for channel, permutation in enumerate(range(3)):
        for term in SECOND_GEN_X_COMBINATIONS:
            base_link = _link_from_key(cyclic_permutation(int(term["link"]), permutation))
            delayed_links = tuple(
                _link_from_key(cyclic_permutation(int(link), permutation))
                for link in term["links_for_delay"]
            )
            delay_s = sum(light_times_s[link] for link in delayed_links)
            phase = np.exp(-2j * np.pi * frequency_hz * delay_s)
            coefficients[:, channel, _LINK_INDEX[base_link]] += float(term["sign"]) * phase
    return coefficients


def _covariance_from_primitives(
    coefficients: np.ndarray,
    link_map: np.ndarray,
    oms_psd: np.ndarray,
    tm_psd: np.ndarray,
) -> np.ndarray:
    primitive_psd = np.concatenate((oms_psd, tm_psd), axis=0).T
    transfer = np.einsum("fcl,flp->fcp", coefficients, link_map, optimize=True)
    return np.einsum("fcp,fp,fdp->fcd", transfer, primitive_psd, transfer.conj(), optimize=True)


def _covariance_from_links(
    coefficients: np.ndarray,
    link_map: np.ndarray,
    oms_psd: np.ndarray,
    tm_psd: np.ndarray,
) -> np.ndarray:
    primitive_psd = np.concatenate((oms_psd, tm_psd), axis=0).T
    link_covariance = np.einsum(
        "flp,fp,fmp->flm", link_map, primitive_psd, link_map.conj(), optimize=True
    )
    return np.einsum(
        "fcl,flm,fdm->fcd", coefficients, link_covariance, coefficients.conj(), optimize=True
    )


def frozen_tdi2_noise_covariance(
    frequency_hz: float | np.ndarray,
    light_times_s: Mapping[Any, float],
    *,
    oms_psd: Any,
    tm_psd: Any,
    basis: str = "AET",
    method: str = "primitive",
) -> np.ndarray:
    """Return a frozen unequal-arm TDI2 instrumental-noise CSD matrix.

    ``light_times_s`` maps directed links ``(i, j)`` to the travel time for
    light propagating from spacecraft ``j`` to ``i``. ``oms_psd`` and
    ``tm_psd`` are fractional-frequency one-sided PSDs. Each may be a common
    scalar/array or a mapping of all six directed links to scalar/arrays.

    The return shape is ``(n_frequency, 3, 3)`` in ``XYZ`` or ``AET`` basis,
    or ``(3, 3)`` for scalar frequency input. The default primitive-noise path
    is numerically stable; ``method='links'`` is an algebraically independent
    link-CSD contraction retained for validation.
    """

    frequency, scalar = _frequency_array(frequency_hz)
    delays = _light_time_map(light_times_s)
    oms = _link_psds(oms_psd, len(frequency), name="oms_psd")
    tm = _link_psds(tm_psd, len(frequency), name="tm_psd")
    coefficients = _xyz_coefficients(frequency, delays)
    link_map = _cleaned_link_map(_delay_phases(frequency, delays))

    method_key = str(method).strip().lower()
    if method_key == "primitive":
        covariance = _covariance_from_primitives(coefficients, link_map, oms, tm)
    elif method_key == "links":
        covariance = _covariance_from_links(coefficients, link_map, oms, tm)
    else:
        raise ValueError("method must be 'primitive' or 'links'")

    basis_key = str(basis).strip().upper()
    if basis_key == "AET":
        covariance = np.einsum(
            "ac,fcd,bd->fab", _AET_FROM_XYZ, covariance, _AET_FROM_XYZ, optimize=True
        )
    elif basis_key != "XYZ":
        raise ValueError("basis must be 'XYZ' or 'AET'")
    return covariance[0] if scalar else covariance


def frozen_tdi2_detector_noise_covariance(
    frequency_hz: float | np.ndarray,
    light_times_s: Mapping[Any, float],
    detector: str | DetectorNoiseModel = "lisa",
    *,
    basis: str = "AET",
    method: str = "primitive",
) -> np.ndarray:
    """Return frozen TDI2 covariance using a built-in detector-noise model."""

    frequency, _scalar = _frequency_array(frequency_hz)
    tm_psd, oms_psd = one_way_noise_psd(frequency, detector)
    return frozen_tdi2_noise_covariance(
        frequency_hz,
        light_times_s,
        oms_psd=oms_psd,
        tm_psd=tm_psd,
        basis=basis,
        method=method,
    )


def frozen_tdi2_t_low_frequency_leakage(
    frequency_hz: float | np.ndarray,
    light_times_s: Mapping[Any, float],
    *,
    oms_psd: Any,
    tm_psd: Any,
) -> np.ndarray:
    """Return the leading reciprocal unequal-arm low-frequency TDI2 leakage.

    This is the leading ``f^4`` term for equal OMS/TM noise levels and is a
    regression diagnostic, not a replacement for the complete covariance.
    """

    frequency, scalar = _frequency_array(frequency_hz)
    delays = _light_time_map(light_times_s)
    for receiver, emitter in _LINKS:
        if not np.isclose(delays[(receiver, emitter)], delays[(emitter, receiver)], rtol=1.0e-12, atol=0.0):
            raise ValueError("the low-frequency leakage formula requires reciprocal light times")
    oms = _link_psds(oms_psd, len(frequency), name="oms_psd")
    tm = _link_psds(tm_psd, len(frequency), name="tm_psd")
    if not np.array_equal(oms, np.broadcast_to(oms[0], oms.shape)):
        raise ValueError("the low-frequency leakage formula requires equal OMS PSDs")
    if not np.array_equal(tm, np.broadcast_to(tm[0], tm.shape)):
        raise ValueError("the low-frequency leakage formula requires equal TM PSDs")

    tau_1 = delays[(2, 3)]
    tau_2 = delays[(1, 3)]
    tau_3 = delays[(1, 2)]
    sigma = tau_1 + tau_2 + tau_3
    mismatch = ((tau_1 - tau_2) ** 2 + (tau_2 - tau_3) ** 2 + (tau_3 - tau_1) ** 2) / 2.0
    omega = 2.0 * np.pi * frequency
    result = (64.0 / 3.0) * omega**4 * sigma**2 * mismatch * (oms[0] + 4.0 * tm[0])
    return result[0] if scalar else result


__all__ = [
    "frozen_tdi2_detector_noise_covariance",
    "frozen_tdi2_light_times_from_positions",
    "frozen_tdi2_noise_covariance",
    "frozen_tdi2_t_low_frequency_leakage",
]
