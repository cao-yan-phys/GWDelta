"""Numba CUDA waveform kernels for the eccentric-GB project."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from .array_backend import ArrayBackend
from .waveforms import WaveformSamples


def _require_numba_cuda():
    """Import ``numba.cuda`` with the project-required binding setting."""

    if os.environ.get("NUMBA_CUDA_USE_NVIDIA_BINDING") != "1":
        raise RuntimeError(
            "Set NUMBA_CUDA_USE_NVIDIA_BINDING=1 before using the numba.cuda backend. "
            "See NUMBA_CUDA_SETUP.md."
        )
    try:
        from numba import cuda
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"numba.cuda is not importable: {exc}") from exc
    if not cuda.is_available():
        raise RuntimeError("numba.cuda reports no available CUDA device")
    return cuda


def _host_backend(reason: str) -> ArrayBackend:
    return ArrayBackend("numpy", np, False, reason)


def _blocks(n: int, threads: int = 256) -> tuple[int, int]:
    return (max(1, (int(n) + threads - 1) // threads), threads)


def _compile_kernels():
    from numba import cuda

    @cuda.jit
    def fixed_pn_qk_kernel(
        t,
        h_plus,
        h_cross,
        m_total,
        nu,
        distance,
        x0,
        e_t,
        l0,
        lambda0,
        t0,
        theta_obs,
        phi_obs,
        include_h20,
        global_sign,
    ):
        i = cuda.grid(1)
        if i >= t.size:
            return

        pi = math.pi
        twopi = 2.0 * pi
        omega = x0 * math.sqrt(x0) / m_total
        k = 3.0 * x0 / (1.0 - e_t * e_t)
        mean_motion = omega / (1.0 + k)
        dt = t[i] - t0
        l = l0 + mean_motion * dt
        lambda_phase = lambda0 + omega * dt

        l_red = math.fmod(l + pi, twopi)
        if l_red < 0.0:
            l_red += twopi
        l_red -= pi
        branch = l - l_red
        u = l_red + e_t * math.sin(l_red) + 0.5 * e_t * e_t * math.sin(2.0 * l_red)
        if e_t > 0.8 and abs(l_red) < 0.1:
            denom = 1.0 - e_t
            if denom < 1.0e-12:
                denom = 1.0e-12
            u = l_red / denom
        for _ in range(30):
            delta = (u - e_t * math.sin(u) - l_red) / (1.0 - e_t * math.cos(u))
            u -= delta
            if abs(delta) < 1.0e-13:
                break
        u += branch

        u_red = math.fmod(u + pi, twopi)
        if u_red < 0.0:
            u_red += twopi
        u_red -= pi
        u_branch = u - u_red
        e_phi = e_t * (1.0 + (4.0 - nu) * x0)
        v_red = 2.0 * math.atan2(
            math.sqrt(1.0 + e_phi) * math.sin(0.5 * u_red),
            math.sqrt(1.0 - e_phi) * math.cos(0.5 * u_red),
        )
        v = v_red + u_branch
        W = (1.0 + k) * (v - l)
        orbital_phase = lambda_phase + W

        cos_u = math.cos(u)
        sin_u = math.sin(u)
        D = 1.0 - e_t * cos_u
        A0 = global_sign * 8.0 * m_total * nu * x0 * math.sqrt(pi / 5.0) / distance
        h20 = A0 * e_t * cos_u / (math.sqrt(6.0) * D)

        num_re = 2.0 - 2.0 * e_t * e_t - e_t * cos_u + e_t * e_t * cos_u * cos_u
        num_im = 2.0 * e_t * math.sqrt(1.0 - e_t * e_t) * sin_u
        pref = A0 / (2.0 * D * D)
        cphase = math.cos(-2.0 * orbital_phase)
        sphase = math.sin(-2.0 * orbital_phase)
        h22_re = pref * (cphase * num_re - sphase * num_im)
        h22_im = pref * (sphase * num_re + cphase * num_im)

        cos_theta = math.cos(theta_obs)
        sin_theta = math.sin(theta_obs)
        y20 = math.sqrt(15.0 / (32.0 * pi)) * sin_theta * sin_theta
        y22_amp = math.sqrt(5.0 / (64.0 * pi)) * (1.0 + cos_theta) * (1.0 + cos_theta)
        y2m2_amp = math.sqrt(5.0 / (64.0 * pi)) * (1.0 - cos_theta) * (1.0 - cos_theta)
        y22_re = y22_amp * math.cos(2.0 * phi_obs)
        y22_im = y22_amp * math.sin(2.0 * phi_obs)
        y2m2_re = y2m2_amp * math.cos(-2.0 * phi_obs)
        y2m2_im = y2m2_amp * math.sin(-2.0 * phi_obs)

        strain_re = h22_re * y22_re - h22_im * y22_im
        strain_im = h22_re * y22_im + h22_im * y22_re
        strain_re += h22_re * y2m2_re + h22_im * y2m2_im
        strain_im += h22_re * y2m2_im - h22_im * y2m2_re
        if include_h20:
            strain_re += h20 * y20

        h_plus[i] = strain_re
        h_cross[i] = -strain_im

    @cuda.jit
    def orbit_polarization_kernel(
        r,
        theta,
        rdot,
        thetadot,
        h_plus,
        h_cross,
        mass,
        nu,
        phis,
        thetas,
    ):
        i = cuda.grid(1)
        if i >= r.size:
            return

        ri = r[i]
        th = theta[i]
        rd = rdot[i]
        thd = thetadot[i]

        sin_th = math.sin(th)
        cos_th = math.cos(th)
        sin_phis = math.sin(phis)
        cos_phis = math.cos(phis)
        cos_thetas = math.cos(thetas)
        cos_thetas2 = cos_thetas * cos_thetas

        radial_projected = sin_th * rd + cos_th * ri * thd
        angular_projected = cos_th * rd - ri * sin_th * thd

        hp = 2.0 * mass * nu * (
            (-cos_phis * cos_phis + cos_thetas2 * sin_phis * sin_phis)
            * (-(mass * sin_th * sin_th / ri) + radial_projected * radial_projected)
            + (1.0 + cos_thetas2)
            * cos_phis
            * sin_phis
            * (-(mass * math.sin(2.0 * th) / ri) + 2.0 * radial_projected * angular_projected)
            + (cos_thetas2 * cos_phis * cos_phis - sin_phis * sin_phis)
            * (-(mass * cos_th * cos_th / ri) + angular_projected * angular_projected)
        )

        phase = phis - th
        hc = (
            2.0
            * mass
            * nu
            * cos_thetas
            * (
                mass * math.sin(2.0 * phase)
                - ri * math.sin(2.0 * phase) * rd * rd
                + 2.0 * math.cos(2.0 * phase) * ri * ri * rd * thd
                + ri * ri * ri * math.sin(2.0 * phase) * thd * thd
            )
            / ri
        )

        h_plus[i] = hp
        h_cross[i] = hc

    return fixed_pn_qk_kernel, orbit_polarization_kernel


_KERNELS = None


def _kernels():
    global _KERNELS
    _require_numba_cuda()
    if _KERNELS is None:
        _KERNELS = _compile_kernels()
    return _KERNELS


def sample_fixed_pn_qk_cuda(
    t,
    params: Any,
    theta: float,
    phi: float,
    *,
    include_h20: bool = True,
) -> WaveformSamples:
    """Sample fixed-parameter PN/QK polarizations with a Numba CUDA kernel."""

    cuda = _require_numba_cuda()
    fixed_kernel, _orbit_kernel = _kernels()
    t_np = np.ascontiguousarray(t, dtype=np.float64)
    hp_np = np.empty_like(t_np)
    hc_np = np.empty_like(t_np)

    d_t = cuda.to_device(t_np)
    d_hp = cuda.device_array_like(t_np)
    d_hc = cuda.device_array_like(t_np)
    fixed_kernel[_blocks(t_np.size)](
        d_t,
        d_hp,
        d_hc,
        float(params.total_mass),
        float(params.nu),
        float(params.distance),
        float(params.x0),
        float(params.e_t0),
        float(params.l0),
        float(params.lambda0),
        float(params.t0),
        float(theta),
        float(phi),
        bool(include_h20),
        float(params.global_sign),
    )
    cuda.synchronize()
    d_hp.copy_to_host(hp_np)
    d_hc.copy_to_host(hc_np)

    omega = params.x0**1.5 / params.total_mass
    k = 3.0 * params.x0 / (1.0 - params.e_t0**2)
    return WaveformSamples(
        t=t_np,
        h_plus=hp_np,
        h_cross=hc_np,
        backend=_host_backend("computed by numba.cuda and copied to host"),
        metadata={
            "model": "PN_QK_0PN_modes_fixed_numba_cuda",
            "backend": "numba_cuda",
            "omega": omega,
            "mean_motion": omega / (1.0 + k),
            "k": k,
        },
    )


def polarizations_from_orbit_state_cuda(r, theta, rdot, thetadot, params: Any) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the notebook quadrupole polarizations with a Numba CUDA kernel."""

    cuda = _require_numba_cuda()
    _fixed_kernel, orbit_kernel = _kernels()
    r_np = np.ascontiguousarray(r, dtype=np.float64)
    theta_np = np.ascontiguousarray(theta, dtype=np.float64)
    rdot_np = np.ascontiguousarray(rdot, dtype=np.float64)
    thetadot_np = np.ascontiguousarray(thetadot, dtype=np.float64)
    if not (r_np.shape == theta_np.shape == rdot_np.shape == thetadot_np.shape):
        raise ValueError("orbit state arrays must have identical shapes")

    hp_np = np.empty_like(r_np)
    hc_np = np.empty_like(r_np)
    d_hp = cuda.device_array_like(r_np)
    d_hc = cuda.device_array_like(r_np)
    orbit_kernel[_blocks(r_np.size)](
        cuda.to_device(r_np),
        cuda.to_device(theta_np),
        cuda.to_device(rdot_np),
        cuda.to_device(thetadot_np),
        d_hp,
        d_hc,
        float(params.total_mass),
        float(params.nu),
        float(params.phis),
        float(params.thetas),
    )
    cuda.synchronize()
    d_hp.copy_to_host(hp_np)
    d_hc.copy_to_host(hc_np)
    return hp_np, hc_np


def sample_accurate_from_solution_cuda(t, solution) -> WaveformSamples:
    """Sample an existing numerical orbit and project it with Numba CUDA."""

    t_np = np.ascontiguousarray(t, dtype=np.float64)
    state = solution.state_at(t_np)
    hp, hc = polarizations_from_orbit_state_cuda(state[0], state[1], state[2], state[3], solution.params)
    return WaveformSamples(
        t=t_np,
        h_plus=hp,
        h_cross=hc,
        backend=_host_backend("computed by numba.cuda and copied to host"),
        metadata={"model": "accurate_numerical_orbit_numba_cuda_projection", "backend": "numba_cuda"},
    )
