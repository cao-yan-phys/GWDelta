"""Frequency-domain static equal-arm Taiji response helpers.

This module contains the project's own analytic static-Taiji one-way response
and first-/second-generation TDI assembly.  It is intentionally independent of the
parameter-estimation demos so production PE code can import the same response
functions used by the FD/TD validation scripts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


C_SI = 299_792_458.0
TAIJI_ARM_M = 3.0e9
DEFAULT_LINKS = (12, 23, 31, 13, 32, 21)


def static_taiji_positions(arm_m: float = TAIJI_ARM_M) -> np.ndarray:
    """Return a static equilateral Taiji triangle centered at the SSB origin."""

    radius = float(arm_m) / np.sqrt(3.0)
    angles = np.deg2rad([0.0, 120.0, 240.0])
    x = np.zeros((3, 3), dtype=float)
    x[:, 0] = radius * np.cos(angles)
    x[:, 1] = radius * np.sin(angles)
    return x


def make_static_taiji_orbits(duration_s: float, *, force_backend: str | None = None):
    """Build a constant-orbit object matching :func:`static_taiji_positions`."""

    from .orbits import SampledOrbits

    # CubicSpline accepts two points and returns a constant orbit for identical
    # endpoint positions.
    t_orbit = np.asarray([0.0, float(duration_s)], dtype=float)
    x0 = static_taiji_positions()
    x = np.repeat(x0[np.newaxis, :, :], len(t_orbit), axis=0)
    return SampledOrbits(t_orbit, x, armlength=TAIJI_ARM_M, force_backend=force_backend)


def sky_basis(lam: float, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return propagation and polarization basis vectors for ecliptic sky angles."""

    cosbeta = np.cos(beta)
    sinbeta = np.sin(beta)
    coslam = np.cos(lam)
    sinlam = np.sin(lam)
    v = np.asarray([-sinbeta * coslam, -sinbeta * sinlam, cosbeta], dtype=float)
    u = np.asarray([sinlam, -coslam, 0.0], dtype=float)
    k = np.asarray([-cosbeta * coslam, -cosbeta * sinlam, -sinbeta], dtype=float)
    return k, u, v


def receivers_emitters(links: Iterable[int] = DEFAULT_LINKS) -> tuple[np.ndarray, np.ndarray]:
    """Map link labels such as ``12`` to receiver/emitter spacecraft indices."""

    links_tuple = tuple(int(link) for link in links)
    receivers = np.asarray([int(str(link)[0]) - 1 for link in links_tuple], dtype=int)
    emitters = np.asarray([int(str(link)[1]) - 1 for link in links_tuple], dtype=int)
    return receivers, emitters


def link_fd_response(
    freqs: np.ndarray,
    *,
    lam: float,
    beta: float,
    positions_m: np.ndarray,
    links: Iterable[int] = DEFAULT_LINKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return link transfer arrays multiplying ``H_plus(f)`` and ``H_cross(f)``.

    The one-way sign follows the current local FastLISAResponse convention
    audited in this project: ``h_em - h_rec``.
    """

    freqs = np.asarray(freqs, dtype=float)
    links_tuple = tuple(int(link) for link in links)
    positions_m = np.asarray(positions_m, dtype=float)
    if positions_m.shape != (3, 3):
        raise ValueError("positions_m must have shape (3, 3)")

    k, u, v = sky_basis(lam, beta)
    receivers, emitters = receivers_emitters(links_tuple)
    x_rec = positions_m[receivers]
    x_em = positions_m[emitters]
    delta = x_rec - x_em
    arm_m = np.linalg.norm(delta, axis=1)
    n = delta / arm_m[:, None]
    arm_s = arm_m / C_SI
    k_dot_n = n @ k
    kx_rec_s = x_rec @ k / C_SI
    kx_em_s = x_em @ k / C_SI
    u_dot_n = n @ u
    v_dot_n = n @ v
    xi_plus = 0.5 * (u_dot_n * u_dot_n - v_dot_n * v_dot_n)
    xi_cross = u_dot_n * v_dot_n

    phase_em = np.exp(-2j * np.pi * freqs[np.newaxis, :] * (arm_s[:, np.newaxis] + kx_em_s[:, np.newaxis]))
    phase_rec = np.exp(-2j * np.pi * freqs[np.newaxis, :] * kx_rec_s[:, np.newaxis])
    denom = (1.0 - k_dot_n)[:, np.newaxis]
    g = (phase_em - phase_rec) / denom
    return xi_plus[:, np.newaxis] * g, xi_cross[:, np.newaxis] * g


def cyclic_permutation(link: int, permutation: int) -> int:
    """Apply the spacecraft cycle 1->2->3->1 to a link label."""

    out = ""
    for ch in str(int(link)):
        sc = int(ch) + int(permutation)
        if sc > 3:
            sc = sc % 3
        out += str(sc)
    return int(out)


FIRST_GEN_X_COMBINATIONS = (
    {"link": 13, "links_for_delay": (), "sign": +1.0},
    {"link": 31, "links_for_delay": (13,), "sign": +1.0},
    {"link": 12, "links_for_delay": (13, 31), "sign": +1.0},
    {"link": 21, "links_for_delay": (13, 31, 12), "sign": +1.0},
    {"link": 12, "links_for_delay": (), "sign": -1.0},
    {"link": 21, "links_for_delay": (12,), "sign": -1.0},
    {"link": 13, "links_for_delay": (12, 21), "sign": -1.0},
    {"link": 31, "links_for_delay": (12, 21, 13), "sign": -1.0},
)


SECOND_GEN_X_COMBINATIONS = FIRST_GEN_X_COMBINATIONS + (
    {"link": 12, "links_for_delay": (13, 31, 12, 21), "sign": +1.0},
    {"link": 21, "links_for_delay": (13, 31, 12, 21, 12), "sign": +1.0},
    {"link": 13, "links_for_delay": (13, 31, 12, 21, 12, 21), "sign": +1.0},
    {"link": 31, "links_for_delay": (13, 31, 12, 21, 12, 21, 13), "sign": +1.0},
    {"link": 13, "links_for_delay": (12, 21, 13, 31), "sign": -1.0},
    {"link": 31, "links_for_delay": (12, 21, 13, 31, 13), "sign": -1.0},
    {"link": 12, "links_for_delay": (12, 21, 13, 31, 13, 31), "sign": -1.0},
    {"link": 21, "links_for_delay": (12, 21, 13, 31, 13, 31, 12), "sign": -1.0},
)


def normalize_tdi_generation(tdi: str) -> str:
    """Normalize user-facing TDI generation labels."""

    key = str(tdi).strip().lower().replace("_", " ").replace("-", " ")
    if key in {"1", "1st", "first", "1st generation", "first generation"}:
        return "1st generation"
    if key in {"2", "2nd", "second", "2nd generation", "second generation"}:
        return "2nd generation"
    raise ValueError("tdi must be '1st generation' or '2nd generation'")


def x_combinations_for_tdi(tdi: str) -> tuple[dict[str, object], ...]:
    """Return the FastLISAResponse-style X-channel delay terms."""

    generation = normalize_tdi_generation(tdi)
    if generation == "1st generation":
        return FIRST_GEN_X_COMBINATIONS
    return SECOND_GEN_X_COMBINATIONS


def xyz_from_link_combinations(
    freqs: np.ndarray,
    y_links_f: np.ndarray,
    *,
    x_combinations: Iterable[dict[str, object]],
    links: Iterable[int] = DEFAULT_LINKS,
    positions_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble cyclic ``X,Y,Z`` from one-way link spectra and X delay terms."""

    freqs = np.asarray(freqs, dtype=float)
    y_links_f = np.asarray(y_links_f, dtype=np.complex128)
    links_tuple = tuple(int(link) for link in links)
    combinations = tuple(x_combinations)
    positions = static_taiji_positions() if positions_m is None else np.asarray(positions_m, dtype=float)
    if positions.shape != (3, 3):
        raise ValueError("positions_m must have shape (3, 3)")

    link_to_index = {link: i for i, link in enumerate(links_tuple)}
    required_links = {
        cyclic_permutation(int(term["link"]), permutation)
        for permutation in range(3)
        for term in combinations
    }
    missing = sorted(required_links - set(link_to_index))
    if missing:
        raise ValueError(f"missing one-way link spectra for links: {missing}")

    receivers, emitters = receivers_emitters(links_tuple)
    arm_s_by_link = {
        link: float(np.linalg.norm(positions[receivers[i]] - positions[emitters[i]]) / C_SI)
        for i, link in enumerate(links_tuple)
    }
    channels = []
    for permutation in range(3):
        channel = np.zeros_like(freqs, dtype=np.complex128)
        for term in combinations:
            base = cyclic_permutation(int(term["link"]), permutation)
            delays = tuple(cyclic_permutation(link, permutation) for link in term["links_for_delay"])
            delay_s = sum(arm_s_by_link[link] for link in delays)
            phase = np.exp(-2j * np.pi * freqs * delay_s)
            channel += float(term["sign"]) * y_links_f[link_to_index[base]] * phase
        channels.append(channel)
    return channels[0], channels[1], channels[2]


def first_generation_xyz_from_links(
    freqs: np.ndarray,
    y_links_f: np.ndarray,
    *,
    links: Iterable[int] = DEFAULT_LINKS,
    positions_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble first-generation Michelson ``X,Y,Z`` from one-way link spectra."""

    return xyz_from_link_combinations(
        freqs,
        y_links_f,
        x_combinations=FIRST_GEN_X_COMBINATIONS,
        links=links,
        positions_m=positions_m,
    )


def second_generation_xyz_from_links(
    freqs: np.ndarray,
    y_links_f: np.ndarray,
    *,
    links: Iterable[int] = DEFAULT_LINKS,
    positions_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble second-generation Michelson ``X,Y,Z`` from one-way link spectra."""

    return xyz_from_link_combinations(
        freqs,
        y_links_f,
        x_combinations=SECOND_GEN_X_COMBINATIONS,
        links=links,
        positions_m=positions_m,
    )


def tdi_xyz_from_links(
    freqs: np.ndarray,
    y_links_f: np.ndarray,
    *,
    tdi: str = "1st generation",
    links: Iterable[int] = DEFAULT_LINKS,
    positions_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``X,Y,Z`` for the requested static TDI generation."""

    return xyz_from_link_combinations(
        freqs,
        y_links_f,
        x_combinations=x_combinations_for_tdi(tdi),
        links=links,
        positions_m=positions_m,
    )


def aet_from_xyz(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the local FastLISAResponse ``A,E,T`` basis from ``X,Y,Z``."""

    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    return A, E, T


@dataclass(frozen=True)
class StaticTaijiFDResponse:
    """Reusable static equal-arm Taiji frequency-domain response calculator."""

    arm_m: float = TAIJI_ARM_M
    links: tuple[int, ...] = DEFAULT_LINKS
    positions_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        links = tuple(int(link) for link in self.links)
        positions = static_taiji_positions(self.arm_m) if self.positions_m is None else np.asarray(self.positions_m, dtype=float)
        if positions.shape != (3, 3):
            raise ValueError("positions_m must have shape (3, 3)")
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "positions_m", positions)

    def link_response(self, freqs: np.ndarray, *, lam: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
        return link_fd_response(freqs, lam=lam, beta=beta, positions_m=self.positions_m, links=self.links)

    def xyz(
        self,
        freqs: np.ndarray,
        h_plus_f: np.ndarray,
        h_cross_f: np.ndarray,
        *,
        lam: float,
        beta: float,
        tdi: str = "1st generation",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hp_resp, hc_resp = self.link_response(freqs, lam=lam, beta=beta)
        y_links = hp_resp * h_plus_f[np.newaxis, :] + hc_resp * h_cross_f[np.newaxis, :]
        return tdi_xyz_from_links(freqs, y_links, tdi=tdi, links=self.links, positions_m=self.positions_m)

    def aet(
        self,
        freqs: np.ndarray,
        h_plus_f: np.ndarray,
        h_cross_f: np.ndarray,
        *,
        lam: float,
        beta: float,
        tdi: str = "1st generation",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, Z = self.xyz(freqs, h_plus_f, h_cross_f, lam=lam, beta=beta, tdi=tdi)
        return aet_from_xyz(X, Y, Z)

    def ae(
        self,
        freqs: np.ndarray,
        h_plus_f: np.ndarray,
        h_cross_f: np.ndarray,
        *,
        lam: float,
        beta: float,
        tdi: str = "1st generation",
    ) -> tuple[np.ndarray, np.ndarray]:
        A, E, _T = self.aet(freqs, h_plus_f, h_cross_f, lam=lam, beta=beta, tdi=tdi)
        return A, E

    def make_orbits(self, duration_s: float, *, force_backend: str | None = None):
        return make_static_taiji_orbits(duration_s, force_backend=force_backend)


def fd_static_taiji_ae(
    freqs: np.ndarray,
    h_plus_f: np.ndarray,
    h_cross_f: np.ndarray,
    *,
    lam: float,
    beta: float,
    arm_m: float = TAIJI_ARM_M,
    positions_m: np.ndarray | None = None,
    tdi: str = "1st generation",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning static Taiji ``A,E``."""

    return StaticTaijiFDResponse(arm_m=arm_m, positions_m=positions_m).ae(
        freqs,
        h_plus_f,
        h_cross_f,
        lam=lam,
        beta=beta,
        tdi=tdi,
    )


__all__ = [
    "C_SI",
    "DEFAULT_LINKS",
    "FIRST_GEN_X_COMBINATIONS",
    "SECOND_GEN_X_COMBINATIONS",
    "TAIJI_ARM_M",
    "StaticTaijiFDResponse",
    "aet_from_xyz",
    "cyclic_permutation",
    "fd_static_taiji_ae",
    "first_generation_xyz_from_links",
    "link_fd_response",
    "make_static_taiji_orbits",
    "normalize_tdi_generation",
    "receivers_emitters",
    "second_generation_xyz_from_links",
    "sky_basis",
    "static_taiji_positions",
    "tdi_xyz_from_links",
    "x_combinations_for_tdi",
    "xyz_from_link_combinations",
]
