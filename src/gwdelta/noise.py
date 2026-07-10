"""Instrumental-noise PSD helpers for simple equal-arm detector models.

The functions in this module use one-sided power spectral densities in
fractional-frequency units, matching the TDI convention used by
``fastlisaresponse``/``lisatools`` and the local memory-analysis notebooks.

Implemented scope:

* instrumental noise only; no Galactic confusion foreground is added;
* equal-arm, static-orbit transfer functions for the ordinary Michelson
  channels and their A/E/T rotations;
* first-generation and second-generation TDI PSDs in the static equal-arm
  approximation.

Formula and parameter sources used for cross-checking:

* A/E/T optimal-channel definitions: Prince et al., Phys. Rev. D 66, 122002
  (2002), https://arxiv.org/abs/gr-qc/0209039.
* LISA single-link OMS/acceleration noise and equal-arm TDI-1/TDI-2 transfer
  checks: Babak et al., LISA-LCST-SGS-TN-001 (2021),
  https://arxiv.org/abs/2108.01167.
* Taiji design parameters: Hu & Wu, Natl. Sci. Rev. 4, 685 (2017),
  https://doi.org/10.1093/nsr/nwx116; the same 3e9 m, 8 pm, 3 fm values are
  quoted in https://arxiv.org/abs/1807.09495.
* TianQin design parameters: Luo et al., Class. Quantum Grav. 33, 035010
  (2016), https://arxiv.org/abs/1512.02076.
* BBO stage-I noise model: Corbin & Cornish, Phys. Rev. D 73, 023001 (2006),
  https://arxiv.org/abs/gr-qc/0512039; see also Crowder & Cornish,
  https://arxiv.org/abs/gr-qc/0506015.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .fd_response import C_SI


@dataclass(frozen=True)
class DetectorNoiseModel:
    """Instrumental-noise constants for an equal-arm detector model.

    Parameters are SI amplitudes.  ``s_acc_m_s2_per_sqrt_hz`` and
    ``s_oms_m_per_sqrt_hz`` are unused for BBO, which follows the special
    stage-I BBO PSD expressions in Corbin & Cornish.
    """

    name: str
    arm_length_m: float
    s_acc_m_s2_per_sqrt_hz: float | None
    s_oms_m_per_sqrt_hz: float | None
    references: tuple[str, ...]
    bbo_stage1: bool = False


NOISE_MODELS: dict[str, DetectorNoiseModel] = {
    "lisa": DetectorNoiseModel(
        name="LISA",
        arm_length_m=2.5e9,
        s_acc_m_s2_per_sqrt_hz=3.0e-15,
        s_oms_m_per_sqrt_hz=15.0e-12,
        references=(
            "Babak et al. 2021, arXiv:2108.01167",
            "Prince et al. 2002, arXiv:gr-qc/0209039",
        ),
    ),
    "taiji": DetectorNoiseModel(
        name="Taiji",
        arm_length_m=3.0e9,
        s_acc_m_s2_per_sqrt_hz=3.0e-15,
        s_oms_m_per_sqrt_hz=8.0e-12,
        references=(
            "Hu & Wu 2017, DOI:10.1093/nsr/nwx116",
            "Taiji source paper parameters, arXiv:1807.09495",
        ),
    ),
    "tianqin": DetectorNoiseModel(
        name="TianQin",
        arm_length_m=0.17e9,
        s_acc_m_s2_per_sqrt_hz=1.0e-15,
        s_oms_m_per_sqrt_hz=1.0e-12,
        references=("Luo et al. 2016, arXiv:1512.02076",),
    ),
    "bbo": DetectorNoiseModel(
        name="BBO stage I",
        arm_length_m=0.05e9,
        s_acc_m_s2_per_sqrt_hz=None,
        s_oms_m_per_sqrt_hz=None,
        references=(
            "Corbin & Cornish 2006, arXiv:gr-qc/0512039",
            "Crowder & Cornish 2005, arXiv:gr-qc/0506015",
        ),
        bbo_stage1=True,
    ),
}

_MODEL_ALIASES = {
    "bbo stage i": "bbo",
    "bbo stage 1": "bbo",
    "bbo-stage-i": "bbo",
    "bbo-stage-1": "bbo",
    "bbo_stage_i": "bbo",
    "bbo_stage_1": "bbo",
}


def _detector_key(detector: str) -> str:
    key = str(detector).strip().lower().replace("_", " ").replace("-", " ")
    return _MODEL_ALIASES.get(key, key)


def available_noise_models() -> tuple[str, ...]:
    """Return the built-in detector-noise model names."""

    return tuple(NOISE_MODELS)


def get_noise_model(detector: str | DetectorNoiseModel) -> DetectorNoiseModel:
    """Return a built-in or user-supplied detector-noise model."""

    if isinstance(detector, DetectorNoiseModel):
        return detector
    key = _detector_key(detector)
    try:
        return NOISE_MODELS[key]
    except KeyError as exc:
        names = ", ".join(available_noise_models())
        raise ValueError(f"unknown detector noise model {detector!r}; available: {names}") from exc


def _as_float_frequency(frequency: float | np.ndarray) -> np.ndarray:
    f = np.asarray(frequency, dtype=float)
    if np.any(~np.isfinite(f)):
        raise ValueError("frequency values must be finite")
    return f


def positive_psd_frequency(frequency: float | np.ndarray, *, replacement_hz: float | None = None) -> np.ndarray:
    """Return frequencies with non-positive bins replaced for analytic PSD calls.

    This is useful when constructing an FFT-aligned PSD array: evaluate the
    analytic expression at ``df`` for the DC bin, then set the DC bin to the
    desired convention outside this function.
    """

    f = _as_float_frequency(frequency)
    if replacement_hz is None:
        positive = f[f > 0.0]
        if positive.size == 0:
            raise ValueError("replacement_hz is required when no positive frequency is present")
        replacement_hz = float(np.min(positive))
    if replacement_hz <= 0.0:
        raise ValueError("replacement_hz must be positive")
    return np.where(f > 0.0, f, float(replacement_hz))


def one_way_noise_psd(
    frequency: float | np.ndarray,
    detector: str | DetectorNoiseModel = "taiji",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S_acc, S_oms)`` one-sided PSDs in fractional-frequency units.

    For LISA, Taiji and TianQin this implements

    ``S_oms = (2 pi f s_oms / c)^2 [1 + (2 mHz / f)^4]``

    and

    ``S_acc = (s_acc / (2 pi f c))^2 [1 + (0.4 mHz / f)^2]
    [1 + (f / 8 mHz)^4]``.

    For BBO stage I it implements the Corbin-Cornish model:

    ``S_oms = 2e-34 / (3 L)^2`` and
    ``S_acc = 9e-34 / ((2 pi f)^4 (3 L)^2)``.
    """

    model = get_noise_model(detector)
    f_in = _as_float_frequency(frequency)
    scalar_input = f_in.ndim == 0
    f = np.atleast_1d(f_in)
    s_acc = np.full_like(f, np.inf, dtype=float)
    s_oms = np.full_like(f, np.inf, dtype=float)

    positive = f > 0.0
    if not np.any(positive):
        return (s_acc[0], s_oms[0]) if scalar_input else (s_acc, s_oms)

    fp = f[positive]
    if model.bbo_stage1:
        denom = (3.0 * model.arm_length_m) ** 2
        s_oms[positive] = 2.0e-34 / denom
        s_acc[positive] = 9.0e-34 / ((2.0 * np.pi * fp) ** 4 * denom)
        return (s_acc[0], s_oms[0]) if scalar_input else (s_acc, s_oms)

    if model.s_acc_m_s2_per_sqrt_hz is None or model.s_oms_m_per_sqrt_hz is None:
        raise ValueError(f"{model.name} is missing non-BBO noise amplitudes")

    s_oms[positive] = (
        (2.0 * np.pi * fp * model.s_oms_m_per_sqrt_hz / C_SI) ** 2
        * (1.0 + (2.0e-3 / fp) ** 4)
    )
    s_acc[positive] = (
        (model.s_acc_m_s2_per_sqrt_hz / (2.0 * np.pi * fp * C_SI)) ** 2
        * (1.0 + (0.4e-3 / fp) ** 2)
        * (1.0 + (fp / 8.0e-3) ** 4)
    )
    return (s_acc[0], s_oms[0]) if scalar_input else (s_acc, s_oms)


def _normalize_tdi_generation(tdi_generation: str | int) -> str:
    key = str(tdi_generation).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "1": "first",
        "1.0": "first",
        "1st": "first",
        "1st generation": "first",
        "first": "first",
        "first generation": "first",
        "tdi1": "first",
        "tdi 1": "first",
        "2": "second",
        "2.0": "second",
        "2nd": "second",
        "2nd generation": "second",
        "second": "second",
        "second generation": "second",
        "tdi2": "second",
        "tdi 2": "second",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("tdi_generation must be 'first'/'1' or 'second'/'2'") from exc


def _channel_list(channels: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(channels, str):
        raw = tuple(channels.upper())
    else:
        raw = tuple(str(channel).strip().upper() for channel in channels)
    allowed = {"A", "E", "T"}
    if not raw or any(channel not in allowed for channel in raw):
        raise ValueError("channels must contain only A, E and T")
    return raw


def equal_arm_aet_noise_psd(
    frequency: float | np.ndarray,
    detector: str | DetectorNoiseModel = "taiji",
    *,
    channels: str | Sequence[str] = "AE",
    tdi_generation: str | int = "second",
    floor: float | None = None,
) -> np.ndarray:
    """Return equal-arm A/E/T instrumental-noise PSDs.

    The returned array has shape ``(n_channels, ...)`` where the trailing shape
    matches ``frequency``.  A and E are identical in this equal-arm model.
    Frequencies at or below zero are returned as ``inf``.

    ``tdi_generation="first"`` implements the A1/E1/T1 transfer functions.
    ``tdi_generation="second"`` multiplies those static equal-arm PSDs by
    ``4 sin^2(2u)``, where ``u = 2 pi f L / c``.
    """

    model = get_noise_model(detector)
    f_in = _as_float_frequency(frequency)
    scalar_input = f_in.ndim == 0
    f = np.atleast_1d(f_in)
    channel_order = _channel_list(channels)
    generation = _normalize_tdi_generation(tdi_generation)

    out = np.full((len(channel_order),) + f.shape, np.inf, dtype=float)
    positive = f > 0.0
    if not np.any(positive):
        return out[:, 0] if scalar_input else out

    fp = f[positive]
    u = 2.0 * np.pi * fp * model.arm_length_m / C_SI
    s_acc, s_oms = one_way_noise_psd(fp, model)

    sin_u = np.sin(u)
    cos_u = np.cos(u)
    cos_2u = np.cos(2.0 * u)
    sin_half_u = np.sin(0.5 * u)

    psd_ae = sin_u**2 * (
        16.0 * (3.0 + 2.0 * cos_u + cos_2u) * s_acc
        + 8.0 * (2.0 + cos_u) * s_oms
    )
    psd_t = 32.0 * sin_u**2 * sin_half_u**2 * (
        4.0 * sin_half_u**2 * s_acc + s_oms
    )

    if generation == "second":
        factor = 4.0 * np.sin(2.0 * u) ** 2
        psd_ae = factor * psd_ae
        psd_t = factor * psd_t

    if floor is not None:
        if floor <= 0.0:
            raise ValueError("floor must be positive when supplied")
        psd_ae = np.maximum(psd_ae, float(floor))
        psd_t = np.maximum(psd_t, float(floor))

    for idx, channel in enumerate(channel_order):
        out[idx][positive] = psd_t if channel == "T" else psd_ae
    return out[:, 0] if scalar_input else out


def equal_arm_ae_noise_psd(
    frequency: float | np.ndarray,
    detector: str | DetectorNoiseModel = "taiji",
    *,
    tdi_generation: str | int = "second",
    floor: float | None = None,
) -> np.ndarray:
    """Return equal-arm A/E instrumental-noise PSDs."""

    return equal_arm_aet_noise_psd(
        frequency,
        detector,
        channels="AE",
        tdi_generation=tdi_generation,
        floor=floor,
    )


def infer_frequency_spacing(frequency: np.ndarray) -> float:
    """Infer the positive uniform frequency spacing of a 1D grid."""

    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("frequency must be one-dimensional with at least two bins")
    diffs = np.diff(f)
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        raise ValueError("frequency grid must be strictly increasing somewhere")
    return float(np.median(positive))


def diagonal_inverse_covariance_from_psd(
    psd_channels: np.ndarray,
    df_hz: float | np.ndarray,
    *,
    floor: float = 1e-300,
) -> np.ndarray:
    """Build a diagonal complex-bin inverse covariance from one-sided PSDs.

    For positive-frequency complex FFT bins, the convention used in the local
    Triangle-BBH PE work is ``C_II(f) = PSD_I(f) / (4 df)`` for independent
    channels, hence ``C^{-1}_II(f) = 4 df / PSD_I(f)``.
    """

    psd = np.asarray(psd_channels, dtype=float)
    if psd.ndim != 2:
        raise ValueError("psd_channels must have shape (n_channels, n_freq)")
    if floor <= 0.0:
        raise ValueError("floor must be positive")

    df = np.asarray(df_hz, dtype=float)
    if df.ndim == 0:
        df = np.full(psd.shape[1], float(df))
    if df.shape != (psd.shape[1],):
        raise ValueError("df_hz must be scalar or have shape (n_freq,)")

    inv_cov = np.zeros((psd.shape[1], psd.shape[0], psd.shape[0]), dtype=np.complex128)
    diag = np.arange(psd.shape[0])
    inv_cov[:, diag, diag] = 4.0 * df[:, np.newaxis] / np.maximum(psd.T, floor)
    return inv_cov


def equal_arm_aet_inverse_covariance(
    frequency: np.ndarray,
    detector: str | DetectorNoiseModel = "taiji",
    *,
    channels: str | Sequence[str] = "AE",
    tdi_generation: str | int = "second",
    df_hz: float | np.ndarray | None = None,
    psd_floor: float = 1e-300,
) -> np.ndarray:
    """Build a diagonal inverse covariance from the built-in equal-arm PSD."""

    f = np.asarray(frequency, dtype=float)
    df = infer_frequency_spacing(f) if df_hz is None else df_hz
    psd = equal_arm_aet_noise_psd(
        f,
        detector,
        channels=channels,
        tdi_generation=tdi_generation,
        floor=psd_floor,
    )
    return diagonal_inverse_covariance_from_psd(psd, df, floor=psd_floor)


__all__ = [
    "DetectorNoiseModel",
    "NOISE_MODELS",
    "available_noise_models",
    "diagonal_inverse_covariance_from_psd",
    "equal_arm_ae_noise_psd",
    "equal_arm_aet_inverse_covariance",
    "equal_arm_aet_noise_psd",
    "get_noise_model",
    "infer_frequency_spacing",
    "one_way_noise_psd",
    "positive_psd_frequency",
]
