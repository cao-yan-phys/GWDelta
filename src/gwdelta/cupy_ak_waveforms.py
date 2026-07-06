"""CuPy RawKernel implementation of the notebook AK Fourier waveform."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import jv

from .array_backend import ArrayBackend
from .waveforms import WaveformSamples


@lru_cache(maxsize=1)
def _ak_kernel():
    import cupy as cp

    code = r'''
    extern "C" __global__
    void ak_fourier_kernel(
        const double* t,
        double* h_plus,
        double* h_cross,
        const int nt,
        const int n_max,
        const double* c1,
        const double* c2,
        const double* c3,
        const double omega0,
        const double omega_dot_k,
        const double omega_ddot_phase,
        const double ddot_omega_add,
        const double periastron_time,
        const double precession_phase0,
        const double precession_phase_rate,
        const double amp,
        const double cos_thetas,
        const double sin_thetas2,
        const double cos_thetas2
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= nt) {
            return;
        }

        double ti = t[i];
        double ti2 = ti * ti;
        double ti3 = ti2 * ti;
        const double two_pi = 6.283185307179586476925286766559;
        double precession_phase = remainder(precession_phase0 + precession_phase_rate * ti, two_pi);
        double cos_precession = cos(precession_phase);
        double sin_precession = sin(precession_phase);
        double plus_total = 0.0;
        double cross_total = 0.0;

        for (int j = 0; j < n_max; ++j) {
            double n = (double)(j + 1);
            double harmonic_phase = remainder(
                n * omega0 * (ti - periastron_time)
                + n * 0.5 * omega_dot_k * ti2
                + ddot_omega_add * n * (omega_ddot_phase / 6.0) * ti3,
                two_pi
            );
            double cos_harmonic = cos(harmonic_phase);
            double sin_harmonic = sin(harmonic_phase);

            plus_total +=
                (1.0 + cos_thetas2)
                * (c1[j] * cos_harmonic * cos_precession + c2[j] * sin_harmonic * sin_precession)
                - sin_thetas2 * c3[j] * cos_harmonic;
            cross_total +=
                2.0 * cos_thetas
                * (c1[j] * cos_harmonic * sin_precession - c2[j] * sin_harmonic * cos_precession);
        }

        h_plus[i] = amp * plus_total;
        h_cross[i] = amp * cross_total;
    }
    '''
    return cp.RawKernel(code, "ak_fourier_kernel")


def _coefficients(e_k: float, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c1 = np.empty(n_max, dtype=np.float64)
    c2 = np.empty(n_max, dtype=np.float64)
    c3 = np.empty(n_max, dtype=np.float64)
    sqrt_one_minus_e2 = np.sqrt(1.0 - e_k**2)
    for n in range(1, n_max + 1):
        ne = n * e_k
        j_nm1 = float(jv(n - 1, ne))
        j_n = float(jv(n, ne))
        c1[n - 1] = (
            (1.0 - e_k**2)
            / e_k**2
            * (
                -2.0 * j_n
                + e_k**2 * j_n
                + n * (2.0 * e_k * j_nm1 - 2.0 * e_k**3 * j_nm1 - 2.0 * j_n + 2.0 * e_k**2 * j_n)
            )
        )
        c2[n - 1] = (
            sqrt_one_minus_e2
            * (1.0 - e_k**2)
            / e_k**2
            * (2.0 * e_k * j_nm1 - 2.0 * j_n + n * (-2.0 * j_n + 2.0 * e_k**2 * j_n))
        )
        c3[n - 1] = (1.0 - e_k**2) * j_n
    return c1, c2, c3


def sample_ak_polarizations_fourier_raw_cuda(
    t,
    params,
    *,
    elements=None,
    n_max: int = 10,
    ddot_omega_add: float = 1.0,
) -> WaveformSamples:
    """Sample AK Fourier polarizations with one fused CuPy RawKernel."""

    import cupy as cp
    from notebook_waveforms import (
        _notebook_omega_ddot_phase_factor,
        _omega_dot_newtonian,
        _qk_periastron_precession_rate,
        initial_ak_elements,
        initial_state,
    )

    init = initial_state(params)
    ak = initial_ak_elements(params) if elements is None else elements
    e_k = ak.eccentricity
    if not (0.0 < e_k < 1.0):
        raise ValueError("AK Fourier expression requires 0 < eK < 1")

    t_arr = cp.asarray(t, dtype=cp.float64)
    hp = cp.empty_like(t_arr)
    hc = cp.empty_like(t_arr)
    c1, c2, c3 = _coefficients(float(e_k), int(n_max))
    c1_d = cp.asarray(c1)
    c2_d = cp.asarray(c2)
    c3_d = cp.asarray(c3)

    omega_dot_k = _omega_dot_newtonian(ak.mean_motion, e_k, params.nu, params.boost_factor)
    omega_ddot_phase = _notebook_omega_ddot_phase_factor(init, params)
    precession_rate = params.integer_pn_factor * _qk_periastron_precession_rate(init, params)
    precession_phase0 = 2.0 * (ak.periastron_phase - params.phis + np.pi / 2.0)
    precession_phase_rate = 2.0 * precession_rate
    amp = 2.0 * params.total_mass * params.nu / (ak.semi_major_axis * (1.0 - e_k**2))
    threads = 256
    blocks = max(1, (int(t_arr.size) + threads - 1) // threads)
    _ak_kernel()(
        (blocks,),
        (threads,),
        (
            t_arr,
            hp,
            hc,
            np.int32(t_arr.size),
            np.int32(n_max),
            c1_d,
            c2_d,
            c3_d,
            np.float64(init.omega0),
            np.float64(omega_dot_k),
            np.float64(omega_ddot_phase),
            np.float64(ddot_omega_add),
            np.float64(ak.periastron_time),
            np.float64(precession_phase0),
            np.float64(precession_phase_rate),
            np.float64(amp),
            np.float64(np.cos(params.thetas)),
            np.float64(np.sin(params.thetas) ** 2),
            np.float64(np.cos(params.thetas) ** 2),
        ),
    )
    return WaveformSamples(
        t=t_arr,
        h_plus=hp,
        h_cross=hc,
        backend=ArrayBackend("cupy", cp, True, "computed by fused CuPy RawKernel"),
        metadata={"model": "AK_fourier_raw_cuda", "backend": "cupy_raw", "n_max": n_max},
    )
