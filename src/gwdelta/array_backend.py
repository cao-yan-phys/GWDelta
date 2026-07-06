"""Small NumPy/CuPy backend selector used by waveform sampling code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayBackend:
    """Container for an array module and host/device conversion helpers."""

    name: str
    xp: Any
    is_gpu: bool
    reason: str = ""

    def asarray(self, value, dtype=None):
        return self.xp.asarray(value, dtype=dtype)

    def zeros_like(self, value, dtype=None):
        return self.xp.zeros_like(value, dtype=dtype)

    def asnumpy(self, value):
        if self.is_gpu:
            return self.xp.asnumpy(value)
        return np.asarray(value)


def select_array_backend(prefer_gpu: bool = True, force: str | None = None) -> ArrayBackend:
    """Return a NumPy or CuPy backend.

    Parameters
    ----------
    prefer_gpu:
        Try CuPy first when ``force`` is not set.
    force:
        ``"numpy"``, ``"cpu"``, ``"cupy"``, ``"cuda"``, or ``"gpu"``.
    """

    if force is not None:
        force = force.lower()
    wants_gpu = force in {"cupy", "cuda", "gpu"} or (force is None and prefer_gpu)
    if force in {"numpy", "cpu"}:
        return ArrayBackend("numpy", np, False, "forced numpy backend")

    if wants_gpu:
        try:
            import cupy as cp

            try:
                device_count = cp.cuda.runtime.getDeviceCount()
            except Exception as exc:  # pragma: no cover - depends on CUDA runtime
                return ArrayBackend("numpy", np, False, f"cupy import succeeded, CUDA check failed: {exc}")
            if device_count > 0:
                return ArrayBackend("cupy", cp, True, f"CUDA devices available: {device_count}")
            return ArrayBackend("numpy", np, False, "cupy import succeeded, but no CUDA device was reported")
        except Exception as exc:
            if force in {"cupy", "cuda", "gpu"}:
                raise RuntimeError(f"requested CuPy backend is unavailable: {exc}") from exc
            return ArrayBackend("numpy", np, False, f"cupy unavailable: {exc}")

    return ArrayBackend("numpy", np, False, "default numpy backend")


def infer_backend_from_array(value) -> ArrayBackend:
    """Infer backend from an existing array without importing CuPy eagerly."""

    module = type(value).__module__.split(".", maxsplit=1)[0]
    if module == "cupy":
        import cupy as cp

        return ArrayBackend("cupy", cp, True, "inferred from input array")
    return ArrayBackend("numpy", np, False, "inferred from input array")
