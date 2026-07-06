"""Vectorized eccentric-GB waveform samplers with optional CuPy acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import jv

from .array_backend import ArrayBackend, select_array_backend

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class WaveformSamples:
    """Uniform time-domain polarizations."""

    t: Any
    h_plus: Any
    h_cross: Any | None
    backend: ArrayBackend
    metadata: dict[str, Any]

    def complex_strain(self):
        if self.h_cross is None:
            raise ValueError("h_cross is required for a detector response")
        return self.h_plus + 1j * self.h_cross

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return (
            self.backend.asnumpy(self.t),
            self.backend.asnumpy(self.h_plus),
            None if self.h_cross is None else self.backend.asnumpy(self.h_cross),
        )


def _reduce_angle(xp, angle):
    reduced = (angle + xp.pi) % (2.0 * xp.pi) - xp.pi
    return reduced, angle - reduced


def solve_kepler_newton_backend(l, e_t, backend: ArrayBackend, tol: float = 1e-13, max_iter: int = 30):
    """Solve ``u - e sin(u) = l`` with vectorized NumPy/CuPy Newton steps."""

    xp = backend.xp
    l_arr = xp.asarray(l, dtype=float)
    e_arr = xp.asarray(e_t, dtype=float)
    l_arr, e_arr = xp.broadcast_arrays(l_arr, e_arr)
    l_red, branch = _reduce_angle(xp, l_arr)
    u = l_red + e_arr * xp.sin(l_red) + 0.5 * e_arr**2 * xp.sin(2.0 * l_red)

    high_e = e_arr > 0.8
    if bool(backend.asnumpy(xp.any(high_e))):
        u = xp.where(high_e & (xp.abs(l_red) < 0.1), l_red / xp.maximum(1.0 - e_arr, 1e-12), u)

    for _ in range(max_iter):
        delta = (u - e_arr * xp.sin(u) - l_red) / (1.0 - e_arr * xp.cos(u))
        u = u - delta
        if float(backend.asnumpy(xp.max(xp.abs(delta)))) < tol:
            break
    return u + branch


def qk_angles_backend(
    l,
    x,
    e_t,
    nu: float,
    backend: ArrayBackend,
    *,
    kepler: Literal["newton"] = "newton",
):
    """Return backend arrays ``u, v, k, e_phi, W`` for the 1PN-QK model."""

    if kepler != "newton":
        raise NotImplementedError("the backend sampler currently implements vectorized Newton Kepler solves")
    xp = backend.xp
    l_arr = xp.asarray(l, dtype=float)
    x_arr = xp.asarray(x, dtype=float)
    e_arr = xp.asarray(e_t, dtype=float)
    l_arr, x_arr, e_arr = xp.broadcast_arrays(l_arr, x_arr, e_arr)
    u = solve_kepler_newton_backend(l_arr, e_arr, backend)
    u_red, branch = _reduce_angle(xp, u)
    e_phi = e_arr * (1.0 + (4.0 - nu) * x_arr)
    v_red = 2.0 * xp.arctan2(
        xp.sqrt(1.0 + e_phi) * xp.sin(u_red / 2.0),
        xp.sqrt(1.0 - e_phi) * xp.cos(u_red / 2.0),
    )
    v = v_red + branch
    k = 3.0 * x_arr / (1.0 - e_arr**2)
    W = (1.0 + k) * (v - l_arr)
    return u, v, k, e_phi, W


def polarizations_20_22_from_phase_arrays(
    t,
    l,
    lambda_phase,
    x,
    e_t,
    *,
    nu: float,
    total_mass: float,
    distance: float,
    theta: float,
    phi: float,
    backend: ArrayBackend | None = None,
    include_h20: bool = True,
    global_sign: float = 1.0,
) -> WaveformSamples:
    """Compute 0PN ``h20,h22`` polarizations from QK phase arrays on NumPy/CuPy."""

    backend = select_array_backend() if backend is None else backend
    xp = backend.xp
    t_arr = xp.asarray(t, dtype=float)
    l_arr = xp.asarray(l, dtype=float)
    lambda_arr = xp.asarray(lambda_phase, dtype=float)
    x_arr = xp.asarray(x, dtype=float)
    e_arr = xp.asarray(e_t, dtype=float)

    u, _v, _k, _e_phi, W = qk_angles_backend(l_arr, x_arr, e_arr, nu, backend)
    orbital_phase = lambda_arr + W
    D = 1.0 - e_arr * xp.cos(u)
    A0 = global_sign * 8.0 * total_mass * nu * x_arr * xp.sqrt(xp.pi / 5.0) / distance
    h20 = A0 * e_arr * xp.cos(u) / (xp.sqrt(6.0) * D)
    numerator = (
        2.0
        - 2.0 * e_arr**2
        - e_arr * xp.cos(u)
        + e_arr**2 * xp.cos(u) ** 2
        + 2.0j * e_arr * xp.sqrt(1.0 - e_arr**2) * xp.sin(u)
    )
    h22 = A0 * xp.exp(-2.0j * orbital_phase) * numerator / (2.0 * D**2)

    cos_theta = xp.cos(theta)
    y20 = xp.sqrt(15.0 / (32.0 * xp.pi)) * xp.sin(theta) ** 2
    y22 = xp.sqrt(5.0 / (64.0 * xp.pi)) * (1.0 + cos_theta) ** 2 * xp.exp(2.0j * phi)
    y2m2 = xp.sqrt(5.0 / (64.0 * xp.pi)) * (1.0 - cos_theta) ** 2 * xp.exp(-2.0j * phi)
    strain = h22 * y22 + xp.conjugate(h22) * y2m2
    if include_h20:
        strain = strain + h20 * y20

    return WaveformSamples(
        t=t_arr,
        h_plus=xp.real(strain),
        h_cross=-xp.imag(strain),
        backend=backend,
        metadata={"model": "PN_QK_0PN_modes_from_phase_arrays", "backend": backend.name},
    )


def sample_fixed_pn_qk(
    t,
    params,
    theta: float,
    phi: float,
    *,
    backend: ArrayBackend | None = None,
    include_h20: bool = True,
) -> WaveformSamples:
    """Sample fixed-parameter 1PN-QK phases and 0PN modes on NumPy/CuPy."""

    backend = select_array_backend() if backend is None else backend
    xp = backend.xp
    t_arr = xp.asarray(t, dtype=float)
    omega = params.x0**1.5 / params.total_mass
    k = 3.0 * params.x0 / (1.0 - params.e_t0**2)
    mean_motion = omega / (1.0 + k)
    dt = t_arr - params.t0
    l = params.l0 + mean_motion * dt
    lambda_phase = params.lambda0 + omega * dt
    out = polarizations_20_22_from_phase_arrays(
        t_arr,
        l,
        lambda_phase,
        params.x0,
        params.e_t0,
        nu=params.nu,
        total_mass=params.total_mass,
        distance=params.distance,
        theta=theta,
        phi=phi,
        backend=backend,
        include_h20=include_h20,
        global_sign=params.global_sign,
    )
    out.metadata.update({"omega": omega, "mean_motion": mean_motion, "k": k})
    return out


def sample_evolving_pn_qk(
    t,
    params,
    theta: float,
    phi: float,
    *,
    backend: ArrayBackend | None = None,
    include_h20: bool = True,
    x0_log: float | None = None,
    rtol: float = 1e-10,
    atol: tuple[float, float, float, float] = (1e-13, 1e-13, 1e-10, 1e-10),
) -> WaveformSamples:
    """Use the existing CPU PN evolution, then evaluate the modes on NumPy/CuPy."""

    from pn_qk_waveform import qk_phases_evolving

    backend = select_array_backend() if backend is None else backend
    phases = qk_phases_evolving(t, params, x0_log=x0_log, rtol=rtol, atol=atol)
    out = polarizations_20_22_from_phase_arrays(
        phases.t,
        phases.l,
        phases.lambda_phase,
        phases.x,
        phases.e_t,
        nu=params.nu,
        total_mass=params.total_mass,
        distance=params.distance,
        theta=theta,
        phi=phi,
        backend=backend,
        include_h20=include_h20,
        global_sign=params.global_sign,
    )
    out.metadata.update(
        {
            "model": "PN_QK_evolving_phases_backend_modes",
            "phase_success": phases.success,
            "phase_message": phases.message,
            "phase_nfev": phases.nfev,
        }
    )
    return out


def polarizations_from_orbit_state_backend(r, theta, rdot, thetadot, params, backend: ArrayBackend) -> tuple[Any, Any]:
    """Evaluate the notebook quadrupole polarization formula on NumPy/CuPy."""

    xp = backend.xp
    r = xp.asarray(r, dtype=float)
    theta = xp.asarray(theta, dtype=float)
    rdot = xp.asarray(rdot, dtype=float)
    thetadot = xp.asarray(thetadot, dtype=float)

    mass = params.total_mass
    nu = params.nu
    phis = params.phis
    thetas = params.thetas

    sin_th = xp.sin(theta)
    cos_th = xp.cos(theta)
    sin_phis = xp.sin(phis)
    cos_phis = xp.cos(phis)
    cos_thetas = xp.cos(thetas)

    radial_projected = sin_th * rdot + cos_th * r * thetadot
    angular_projected = cos_th * rdot - r * sin_th * thetadot

    hp = 2.0 * mass * nu * (
        (-cos_phis**2 + cos_thetas**2 * sin_phis**2)
        * (-(mass * sin_th**2 / r) + radial_projected**2)
        + (1.0 + cos_thetas**2)
        * cos_phis
        * sin_phis
        * (-(mass * xp.sin(2.0 * theta) / r) + 2.0 * radial_projected * angular_projected)
        + (cos_thetas**2 * cos_phis**2 - sin_phis**2)
        * (-(mass * cos_th**2 / r) + angular_projected**2)
    )

    phase = phis - theta
    hc = (
        2.0
        * mass
        * nu
        * cos_thetas
        * (
            mass * xp.sin(2.0 * phase)
            - r * xp.sin(2.0 * phase) * rdot**2
            + 2.0 * xp.cos(2.0 * phase) * r**2 * rdot * thetadot
            + r**3 * xp.sin(2.0 * phase) * thetadot**2
        )
        / r
    )
    return hp, hc


def sample_accurate_from_solution(t, solution, *, backend: ArrayBackend | None = None) -> WaveformSamples:
    """Sample an existing numerical orbit and evaluate polarizations on NumPy/CuPy."""

    backend = select_array_backend() if backend is None else backend
    t_np = np.asarray(t, dtype=float)
    state = solution.state_at(t_np)
    hp, hc = polarizations_from_orbit_state_backend(state[0], state[1], state[2], state[3], solution.params, backend)
    return WaveformSamples(
        t=backend.asarray(t_np),
        h_plus=hp,
        h_cross=hc,
        backend=backend,
        metadata={"model": "accurate_numerical_orbit_backend_projection", "backend": backend.name},
    )


def sample_ak_polarizations_fourier(
    t,
    params,
    *,
    elements=None,
    n_max: int = 10,
    ddot_omega_add: float = 1.0,
    backend: ArrayBackend | None = None,
) -> WaveformSamples:
    """Sample the notebook AK Fourier polarizations on NumPy/CuPy."""

    from notebook_waveforms import (
        _notebook_omega_ddot_phase_factor,
        _omega_dot_newtonian,
        _qk_periastron_precession_rate,
        initial_ak_elements,
        initial_state,
    )

    backend = select_array_backend() if backend is None else backend
    xp = backend.xp
    init = initial_state(params)
    ak = initial_ak_elements(params) if elements is None else elements
    e_k = ak.eccentricity
    if not (0.0 < e_k < 1.0):
        raise ValueError("AK Fourier expression requires 0 < eK < 1")

    t_arr = xp.asarray(t, dtype=float)
    plus_total = xp.zeros_like(t_arr, dtype=float)
    cross_total = xp.zeros_like(t_arr, dtype=float)
    mass = params.total_mass
    nu = params.nu
    cos_thetas = xp.cos(params.thetas)
    sin_thetas2 = xp.sin(params.thetas) ** 2
    cos_thetas2 = xp.cos(params.thetas) ** 2

    omega_dot_k = _omega_dot_newtonian(ak.mean_motion, e_k, nu, params.boost_factor)
    omega_ddot_phase = _notebook_omega_ddot_phase_factor(init, params)
    precession_rate = params.integer_pn_factor * _qk_periastron_precession_rate(init, params)
    precession_phase = 2.0 * (ak.periastron_phase + precession_rate * t_arr - params.phis + xp.pi / 2.0)
    precession_phase, _precession_branch = _reduce_angle(xp, precession_phase)
    cos_precession = xp.cos(precession_phase)
    sin_precession = xp.sin(precession_phase)
    sqrt_one_minus_e2 = xp.sqrt(1.0 - e_k**2)

    for n in range(1, n_max + 1):
        ne = n * e_k
        j_nm1 = float(jv(n - 1, ne))
        j_n = float(jv(n, ne))
        harmonic_phase = (
            n * init.omega0 * (t_arr - ak.periastron_time)
            + n * 0.5 * omega_dot_k * t_arr**2
            + ddot_omega_add * n * (omega_ddot_phase / 6.0) * t_arr**3
        )
        harmonic_phase, _harmonic_branch = _reduce_angle(xp, harmonic_phase)
        cos_harmonic = xp.cos(harmonic_phase)
        sin_harmonic = xp.sin(harmonic_phase)
        c1 = (
            (1.0 - e_k**2)
            / e_k**2
            * (
                -2.0 * j_n
                + e_k**2 * j_n
                + n * (2.0 * e_k * j_nm1 - 2.0 * e_k**3 * j_nm1 - 2.0 * j_n + 2.0 * e_k**2 * j_n)
            )
        )
        c2 = (
            sqrt_one_minus_e2
            * (1.0 - e_k**2)
            / e_k**2
            * (2.0 * e_k * j_nm1 - 2.0 * j_n + n * (-2.0 * j_n + 2.0 * e_k**2 * j_n))
        )
        c3 = (1.0 - e_k**2) * j_n
        plus_total += (
            (1.0 + cos_thetas2)
            * (c1 * cos_harmonic * cos_precession + c2 * sin_harmonic * sin_precession)
            - sin_thetas2 * c3 * cos_harmonic
        )
        cross_total += 2.0 * cos_thetas * (
            c1 * cos_harmonic * sin_precession - c2 * sin_harmonic * cos_precession
        )

    amp = 2.0 * mass * nu / (ak.semi_major_axis * (1.0 - e_k**2))
    return WaveformSamples(
        t=t_arr,
        h_plus=amp * plus_total,
        h_cross=amp * cross_total,
        backend=backend,
        metadata={"model": "AK_fourier", "backend": backend.name, "n_max": n_max},
    )


def sample_ak_hplus_fourier(
    t,
    params,
    *,
    elements=None,
    n_max: int = 10,
    ddot_omega_add: float = 1.0,
    backend: ArrayBackend | None = None,
) -> WaveformSamples:
    """Backward-compatible AK sampler name; returns both polarizations."""

    return sample_ak_polarizations_fourier(
        t,
        params,
        elements=elements,
        n_max=n_max,
        ddot_omega_add=ddot_omega_add,
        backend=backend,
    )
