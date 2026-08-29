"""Low-level FastLISAResponse adapter for precomputed h_plus/h_cross arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .array_backend import infer_backend_from_array
from .cuda_runtime import backend_wants_cuda, ensure_cuda_dll_directories
from .tdi_combinations import resolve_tdi_combinations


@dataclass
class TDIResult:
    """Container for raw projections and TDI channels."""

    t: Any
    channels: dict[str, Any]
    projections: Any
    response_model: Any
    metadata: dict[str, Any]

    def as_numpy(self) -> dict[str, np.ndarray]:
        backend = infer_backend_from_array(next(iter(self.channels.values())))
        out = {"t": np.asarray(self.t)}
        for key, value in self.channels.items():
            out[key] = backend.asnumpy(value)
        if self.projections is not None:
            out["projections"] = backend.asnumpy(self.projections)
        return out

    @property
    def link_order(self) -> list[int]:
        return list(self.metadata.get("link_order", []))


def _require_fastlisaresponse():
    try:
        from fastlisaresponse import pyResponseTDI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "FastLISAResponse is not importable. Install the local source tree or a wheel before running TDI."
        ) from exc
    return pyResponseTDI


def _check_uniform_time(t) -> tuple[np.ndarray, float]:
    t_np = np.asarray(t, dtype=float)
    if t_np.ndim != 1 or len(t_np) < 2:
        raise ValueError("t must be a one-dimensional array with at least two samples")
    dt = np.diff(t_np)
    dt0 = float(np.median(dt))
    if not np.allclose(dt, dt0, rtol=1e-10, atol=max(1e-12, abs(dt0) * 1e-12)):
        raise ValueError("FastLISAResponse requires uniformly sampled data")
    return t_np, dt0


class FastLISAResponseTDI:
    """Wrap ``pyResponseTDI`` around already sampled polarizations.

    This class intentionally uses the low-level API, not ``ResponseWrapper``.
    It exposes arm projections, the link order, and the raw response model so
    custom orbit behavior can be inspected.
    """

    def __init__(
        self,
        *,
        orbits=None,
        order: int = 25,
        tdi: str | list[dict[str, Any]] = "1st generation",
        tdi_chan: str = "XYZ",
        force_backend: str | None = None,
        t_buffer: float = 10000.0,
        trim_garbage: bool = False,
        cache_response: bool = True,
    ) -> None:
        self.orbits = orbits
        self.order = int(order)
        self.tdi_requested = tdi
        self.tdi = resolve_tdi_combinations(tdi)
        self.tdi_chan = tdi_chan
        self.force_backend = force_backend
        self.t_buffer = float(t_buffer)
        self.trim_garbage = bool(trim_garbage)
        self.cache_response = bool(cache_response)
        self._cached_response_key: tuple[Any, ...] | None = None
        self._cached_response: Any | None = None

    def clear_response_cache(self) -> None:
        """Drop the cached ``pyResponseTDI`` object."""

        self._cached_response_key = None
        self._cached_response = None

    def _get_response(self, *, dt: float, num_pts: int):
        """Return a cached or newly built ``pyResponseTDI`` instance."""

        if backend_wants_cuda(self.force_backend):
            ensure_cuda_dll_directories()
        pyResponseTDI = _require_fastlisaresponse()
        key = (
            float(dt),
            int(num_pts),
            self.order,
            self.tdi_chan,
            self.force_backend,
            id(self.orbits),
            repr(self.tdi),
        )
        if self.cache_response and self._cached_response_key == key and self._cached_response is not None:
            return self._cached_response, True

        response = pyResponseTDI(
            sampling_frequency=1.0 / dt,
            num_pts=num_pts,
            order=self.order,
            tdi=self.tdi,
            orbits=self.orbits,
            tdi_chan=self.tdi_chan,
            force_backend=self.force_backend,
        )
        if self.cache_response:
            self._cached_response_key = key
            self._cached_response = response
        return response, False

    def compute(
        self,
        t,
        h_plus,
        h_cross,
        *,
        lam: float,
        beta: float,
        t0: float | None = None,
    ) -> TDIResult:
        """Return TDI channels for a precomputed complex strain."""

        t_np, dt = _check_uniform_time(t)
        hp_backend = infer_backend_from_array(h_plus)
        hp = hp_backend.xp.asarray(h_plus)
        hc = hp_backend.xp.asarray(h_cross)
        if hp.shape != hc.shape or hp.shape != t_np.shape:
            raise ValueError("t, h_plus, and h_cross must have the same shape")

        response, cache_hit = self._get_response(dt=dt, num_pts=len(t_np))
        strain = hp + 1j * hc
        start_time = float(t_np[0] if t0 is None else t0)
        response.get_projections(strain, lam, beta, t0=start_time, t_buffer=self.t_buffer)
        channel_values = response.get_tdi_delays()

        if self.tdi_chan == "XYZ":
            names = ["X", "Y", "Z"]
        elif self.tdi_chan == "AET":
            names = ["A", "E", "T"]
        elif self.tdi_chan == "AE":
            names = ["A", "E"]
        else:
            raise ValueError("tdi_chan must be 'XYZ', 'AET', or 'AE'")

        channels = dict(zip(names, channel_values))
        t_out = t_np.copy()
        if self.trim_garbage:
            ind = int(response.tdi_start_ind)
            if 2 * ind >= len(t_out):
                raise ValueError("t_buffer removes all samples; reduce t_buffer or increase the data length")
            t_out = t_out[ind:-ind]
            channels = {key: value[ind:-ind] for key, value in channels.items()}

        return TDIResult(
            t=t_out,
            channels=channels,
            projections=response.y_gw,
            response_model=response,
            metadata={
                "dt": dt,
                "t0": start_time,
                "t_buffer": self.t_buffer,
                "tdi": self.tdi,
                "tdi_requested": self.tdi_requested,
                "tdi_chan": self.tdi_chan,
                "force_backend": self.force_backend,
                "backend": response.backend.name,
                "response_cache_hit": cache_hit,
                "link_order": list(response.response_orbits.LINKS),
                "tdi_start_ind": int(response.tdi_start_ind),
                "projection_start_ind": int(response.projections_start_ind),
                "projection_buffer": int(response.projection_buffer),
            },
        )

    def compute_polarizations(
        self,
        t,
        h_polarizations,
        *,
        lam: float,
        beta: float,
        source_time_s=None,
        reference_position_m=(0.0, 0.0, 0.0),
        quadrature_order: int = 16,
        chunk_size: int = 32768,
        t0: float | None = None,
    ) -> TDIResult:
        """Return TDI for sampled E(2) plane-wave polarizations.

        ``h_polarizations`` maps any nonempty subset of the six public
        polarization names to real time-domain strain arrays. ``source_time_s``
        gives the SSB time grid of those strains and defaults to ``t``. It must
        cover all retarded times sampled along the detector links.
        """

        from .weak_field import PlaneWavePolarizationField, WeakFieldLinkResponse

        t_np, _dt = _check_uniform_time(t)
        field = PlaneWavePolarizationField(
            t_np if source_time_s is None else source_time_s,
            h_polarizations,
            lam=lam,
            beta=beta,
            reference_position_m=reference_position_m,
        )
        link_response = WeakFieldLinkResponse(
            orbits=self.orbits,
            quadrature_order=quadrature_order,
            chunk_size=chunk_size,
            force_backend=self.force_backend,
        )
        links = link_response.compute(t_np, field)
        result = self.compute_links(t_np, links.total, t0=t0)
        result.metadata.update(
            {
                "input_kind": "sampled_plane_wave_polarizations",
                "polarizations": list(field.polarization_names),
                "source_time_start_s": field.time_support_s[0],
                "source_time_stop_s": field.time_support_s[1],
                "reference_position_m": field.reference_position_m.tolist(),
                "link_quadrature_order": int(quadrature_order),
                "link_chunk_size": int(chunk_size),
                "link_response_backend": links.metadata["backend"],
            }
        )
        return result

    def compute_links(self, t, y_links, *, t0: float | None = None) -> TDIResult:
        """Apply the configured TDI delays to precomputed one-way signals.

        ``y_links`` must have shape ``(nlinks, nt)`` and follow the link order
        exposed by ``orbits.LINKS``.  In the local convention, link ``ij`` is
        received at spacecraft ``i`` after emission from spacecraft ``j``.
        """

        t_np, dt = _check_uniform_time(t)
        response, cache_hit = self._get_response(dt=dt, num_pts=len(t_np))
        if int(response.num_pts) != len(t_np):
            raise ValueError("the orbit does not cover the full precomputed-link time grid")

        links = response.xp.asarray(y_links, dtype=response.xp.float64)
        expected_shape = (len(response.response_orbits.LINKS), len(t_np))
        if links.shape != expected_shape:
            raise ValueError(
                f"y_links must have shape {expected_shape} in orbits.LINKS order; got {links.shape}"
            )

        minimum_buffer = 100.0 + 4.0 * self.order * dt
        if self.t_buffer < minimum_buffer:
            raise ValueError(
                "t_buffer is too short for direct-link TDI interpolation; "
                f"use at least {minimum_buffer:g} s"
            )

        start_time = float(t_np[0] if t0 is None else t0)
        response.t0_projection = None
        response.tdi_start_ind = int(self.t_buffer / dt)
        # Some FastLISAResponse releases expose ``y_gw`` in get_tdi_delays()
        # but still validate it through an uninitialized legacy attribute.
        # Populate the same internal projection storage explicitly so the
        # supported TDI kernel is usable across those releases.
        response.nlinks = expected_shape[0]
        response.y_gw_flat = links.flatten().copy()
        response.y_gw_length = len(t_np)
        channel_values = response.get_tdi_delays(t0=start_time)

        if self.tdi_chan == "XYZ":
            names = ["X", "Y", "Z"]
        elif self.tdi_chan == "AET":
            names = ["A", "E", "T"]
        elif self.tdi_chan == "AE":
            names = ["A", "E"]
        else:
            raise ValueError("tdi_chan must be 'XYZ', 'AET', or 'AE'")

        channels = dict(zip(names, channel_values))
        t_out = t_np.copy()
        if self.trim_garbage:
            ind = int(response.tdi_start_ind)
            if 2 * ind >= len(t_out):
                raise ValueError("t_buffer removes all samples; reduce t_buffer or increase the data length")
            t_out = t_out[ind:-ind]
            channels = {key: value[ind:-ind] for key, value in channels.items()}

        return TDIResult(
            t=t_out,
            channels=channels,
            projections=response.y_gw,
            response_model=response,
            metadata={
                "dt": dt,
                "t0": start_time,
                "t_buffer": self.t_buffer,
                "tdi": self.tdi,
                "tdi_requested": self.tdi_requested,
                "tdi_chan": self.tdi_chan,
                "force_backend": self.force_backend,
                "backend": response.backend.name,
                "response_cache_hit": cache_hit,
                "input_kind": "precomputed_links",
                "link_order": list(response.response_orbits.LINKS),
                "tdi_start_ind": int(response.tdi_start_ind),
            },
        )
