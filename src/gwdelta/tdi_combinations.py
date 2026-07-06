"""User-facing TDI delay-combination builders.

The delay-term dictionaries use the local FastLISAResponse convention:
``link`` is the one-way projection label and ``links_for_delay`` is the list of
one-way light-time labels applied to that projection.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any


TDICombination = dict[str, Any]


def _term(link: int, delays: Sequence[int] = (), sign: float = 1.0) -> TDICombination:
    return {"link": int(link), "links_for_delay": tuple(int(delay) for delay in delays), "sign": float(sign)}


# Eq. (2) of Gang Wang, Phys. Rev. D 110, 042005 (2024), mapped into the
# local FastLISAResponse one-way-link convention by reversing every ij label.
FIRST_GEN_RELAY_U_COMBINATIONS: tuple[TDICombination, ...] = (
    _term(31, (), +1.0),
    _term(12, (31,), +1.0),
    _term(23, (31, 12), +1.0),
    _term(32, (31, 12, 23), +1.0),
    _term(32, (), -1.0),
    _term(23, (32,), -1.0),
    _term(31, (32, 23), -1.0),
    _term(12, (32, 23, 31), -1.0),
)


# Eq. (3) of the same paper, again mapped into the local convention.
FIRST_GEN_RELAY_UBAR_COMBINATIONS: tuple[TDICombination, ...] = (
    _term(23, (), +1.0),
    _term(32, (23,), +1.0),
    _term(21, (23, 32), +1.0),
    _term(13, (23, 32, 21), +1.0),
    _term(21, (), -1.0),
    _term(13, (21,), -1.0),
    _term(32, (21, 13), -1.0),
    _term(23, (21, 13, 32), -1.0),
)


# Equal-arm 4L delay used for Ubar in U Ubar.  For time-dependent arms this is
# an experimental concrete path choice corresponding to the first Relay-U beam.
HYBRID_RELAY_UUBAR_DELAY_PATH: tuple[int, ...] = (31, 12, 23, 32)


def _with_extra_delay(
    terms: Iterable[TDICombination],
    extra_delay_path: Sequence[int],
    *,
    sign_scale: float = 1.0,
) -> tuple[TDICombination, ...]:
    prefix = tuple(int(link) for link in extra_delay_path)
    out: list[TDICombination] = []
    for term in terms:
        out.append(
            _term(
                int(term["link"]),
                prefix + tuple(int(link) for link in term["links_for_delay"]),
                float(sign_scale) * float(term["sign"]),
            )
        )
    return tuple(out)


def hybrid_relay_uubar_combinations(
    *,
    extra_delay_path: Sequence[int] = HYBRID_RELAY_UUBAR_DELAY_PATH,
) -> tuple[TDICombination, ...]:
    """Return terms for the hybrid Relay ordinary triplet.

    The first ordinary channel is ``UUbar = U(t) + Ubar(t - 4L)`` in the
    equal-arm limit.  FastLISAResponse cyclically permutes this first-channel
    definition to produce ``VVbar`` and ``WWbar``.
    """

    return FIRST_GEN_RELAY_U_COMBINATIONS + _with_extra_delay(
        FIRST_GEN_RELAY_UBAR_COMBINATIONS,
        extra_delay_path,
        sign_scale=+1.0,
    )


def normalize_tdi_name(tdi: str) -> str:
    """Normalize user-facing TDI preset names."""

    key = str(tdi).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "hybrid relay": "hybrid relay",
        "hybrid relay uubar": "hybrid relay",
        "uubar": "hybrid relay",
        "uu bar": "hybrid relay",
        "uu ubar": "hybrid relay",
        "chinese knot": "hybrid relay",
        "wang gang": "hybrid relay",
        "gang wang": "hybrid relay",
    }
    if key in aliases:
        return aliases[key]
    return str(tdi)


def resolve_tdi_combinations(tdi: str | list[TDICombination]) -> str | list[TDICombination]:
    """Return a pyResponseTDI-compatible TDI specification."""

    if isinstance(tdi, list):
        return deepcopy(tdi)
    if normalize_tdi_name(tdi) == "hybrid relay":
        return [dict(term) for term in hybrid_relay_uubar_combinations()]
    return tdi


def ordinary_channel_names(tdi: str | list[TDICombination]) -> tuple[str, str, str]:
    """Return names for the three ordinary channels before A/E/T rotation."""

    if isinstance(tdi, list):
        return ("C0", "C1", "C2")
    if normalize_tdi_name(tdi) == "hybrid relay":
        return ("UUbar", "VVbar", "WWbar")
    return ("X", "Y", "Z")


__all__ = [
    "FIRST_GEN_RELAY_U_COMBINATIONS",
    "FIRST_GEN_RELAY_UBAR_COMBINATIONS",
    "HYBRID_RELAY_UUBAR_DELAY_PATH",
    "hybrid_relay_uubar_combinations",
    "normalize_tdi_name",
    "ordinary_channel_names",
    "resolve_tdi_combinations",
]
