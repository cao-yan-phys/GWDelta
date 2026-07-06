"""Cached AK -> 2nd-generation AE spectrum calculator for PE loops."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .cuda_runtime import ensure_cuda_dll_directories
from .cupy_ak_waveforms import sample_ak_polarizations_fourier_raw_cuda
from .units import make_physical_scale


@dataclass(frozen=True)
class AESpectrum:
    """Frequency-domain A/E channels returned by ``AKAESpectrumCalculator``."""

    f: Any
    A: Any
    E: Any
    metadata: dict[str, Any]

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cupy as cp

        return cp.asnumpy(self.f), cp.asnumpy(self.A), cp.asnumpy(self.E)


class AKAESpectrumCalculator:
    """Reusable CUDA calculator for AK polarizations, AE TDI, and FFT.

    The expensive response/orbit setup is done once in ``__init__``.  Each
    ``compute`` call only updates the source waveform, detector projection,
    2nd-generation AE TDI, and the final GPU FFT.
    """

    def __init__(
        self,
        *,
        n_samples: int,
        dt: float,
        orbits,
        t_buffer: float = 2400.0,
        order: int = 15,
        n_max: int = 10,
    ) -> None:
        import cupy as cp
        from fastlisaresponse import pyResponseTDI

        ensure_cuda_dll_directories()
        self.cp = cp
        self.n_samples = int(n_samples)
        self.dt = float(dt)
        self.t_buffer = float(t_buffer)
        self.order = int(order)
        self.orbits = orbits
        self.n_max = int(n_max)
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.n_max < 1:
            raise ValueError("n_max must be positive")
        if self.orbits is None:
            raise ValueError("orbits must be supplied explicitly; do not rely on a default orbit in PE code")
        self.response = pyResponseTDI(
            sampling_frequency=1.0 / self.dt,
            num_pts=self.n_samples,
            order=self.order,
            tdi="2nd generation",
            orbits=self.orbits,
            tdi_chan="AE",
            force_backend="cuda12x",
        )
        cp.cuda.Stream.null.synchronize()

        self.t_seconds = cp.arange(self.n_samples, dtype=cp.float64) * self.dt
        self.tdi_start_ind = int(self.t_buffer / self.dt)
        self.n_trim = self.n_samples - 2 * self.tdi_start_ind
        if self.n_trim <= 1:
            raise ValueError("t_buffer removes all usable samples")
        self.f = cp.asarray(np.fft.rfftfreq(self.n_trim, d=self.dt))

    def compute(
        self,
        params,
        *,
        total_mass_solar: float = 1.0e5,
        distance: float = 100.0,
        distance_unit: str = "Mpc",
        lam: float = 0.3,
        beta: float = 0.4,
        copy_to_host: bool = False,
        profile: bool = False,
    ) -> AESpectrum:
        """Return ``dt * rfft(A), dt * rfft(E)`` for one parameter point."""

        cp = self.cp
        timings = {}
        scale = make_physical_scale(
            total_mass_solar=total_mass_solar,
            distance=distance,
            distance_unit=distance_unit,
            code_total_mass=params.total_mass,
        )
        tic = time.perf_counter()
        t_code = self.t_seconds / scale.time_unit_s
        samples = sample_ak_polarizations_fourier_raw_cuda(t_code, params, n_max=self.n_max)
        strain = samples.h_plus * scale.strain_scale + 1j * samples.h_cross * scale.strain_scale
        if profile:
            cp.cuda.Stream.null.synchronize()
            timings["waveform_s"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.response.get_projections(strain, lam, beta, t0=0.0, t_buffer=self.t_buffer)
        if profile:
            cp.cuda.Stream.null.synchronize()
            timings["projection_s"] = time.perf_counter() - tic

        tic = time.perf_counter()
        A_t, E_t = self.response.get_tdi_delays()
        if profile:
            cp.cuda.Stream.null.synchronize()
            timings["tdi_s"] = time.perf_counter() - tic

        ind = self.tdi_start_ind
        tic = time.perf_counter()
        A_f = self.dt * cp.fft.rfft(A_t[ind:-ind])
        E_f = self.dt * cp.fft.rfft(E_t[ind:-ind])
        cp.cuda.Stream.null.synchronize()
        if profile:
            timings["fft_s"] = time.perf_counter() - tic
        metadata = {
            "backend": "cuda12x",
            "model": "AK_fourier_2nd_generation_AE_spectrum",
            "n_samples": self.n_samples,
            "n_trim": self.n_trim,
            "dt": self.dt,
            "df": 1.0 / (self.n_trim * self.dt),
            "n_max": self.n_max,
            "tdi_start_ind": self.tdi_start_ind,
            "time_unit_s": scale.time_unit_s,
            "strain_scale": scale.strain_scale,
        }
        if profile:
            metadata["timings"] = timings
        if copy_to_host:
            return AESpectrum(cp.asnumpy(self.f), cp.asnumpy(A_f), cp.asnumpy(E_f), metadata)
        return AESpectrum(self.f, A_f, E_f, metadata)
