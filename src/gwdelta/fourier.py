from __future__ import annotations

from .array_backend import infer_backend_from_array


def fft(values, n=None, axis=-1, norm=None):
    xp = infer_backend_from_array(values).xp
    return xp.conj(xp.fft.fft(xp.conj(xp.asarray(values)), n=n, axis=axis, norm=norm))


def ifft(values, n=None, axis=-1, norm=None):
    xp = infer_backend_from_array(values).xp
    return xp.conj(xp.fft.ifft(xp.conj(xp.asarray(values)), n=n, axis=axis, norm=norm))


def rfft(values, n=None, axis=-1, norm=None):
    xp = infer_backend_from_array(values).xp
    values = xp.asarray(values)
    if xp.iscomplexobj(values):
        raise ValueError("rfft requires real input; use fft for complex components")
    return xp.conj(xp.fft.rfft(values, n=n, axis=axis, norm=norm))


def irfft(values, n=None, axis=-1, norm=None):
    xp = infer_backend_from_array(values).xp
    return xp.fft.irfft(xp.conj(xp.asarray(values)), n=n, axis=axis, norm=norm)
