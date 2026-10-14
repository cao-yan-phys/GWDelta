import math

import numpy as np

_kernel = None
_weights = {}
_runtime_ready = False


def _ensure_runtime():
    global _runtime_ready
    if not _runtime_ready:
        from .cuda_runtime import ensure_cuda_dll_directories

        ensure_cuda_dll_directories()
        _runtime_ready = True


def _shifted(a, offsets, order):
    global _kernel
    import cupy as cp

    if _kernel is None:
        _ensure_runtime()
        _kernel = cp.RawKernel(r'''
extern "C" __global__ void shift(const double* a, const double* s,
                                 const double* w, double* out, int n, int order) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int k = (int)floor(s[i]);
    double f = s[i] - k;
    if (f == 0.) {
        int q = i + k;
        out[i] = q >= 0 && q < n ? a[q] : 0.;
        return;
    }
    int half = (order + 1) / 2;
    double x = half - 1 + f, numerator = 0., denominator = 0.;
    for (int j = 0; j <= order; ++j) {
        double weight = w[j] / (x - j);
        int q = i + k - (half - 1) + j;
        denominator += weight;
        if (q >= 0 && q < n) numerator += weight * a[q];
    }
    out[i] = numerator / denominator;
}
''', "shift")
    key = (cp.cuda.runtime.getDevice(), order)
    if key not in _weights:
        _weights[key] = cp.asarray([(-1.)**j * math.comb(order, j) for j in range(order + 1)])
    out = cp.empty_like(a)
    _kernel(((a.size + 255) // 256,), (256,), (a, offsets, _weights[key], out, a.size, order))
    return out


def _compute_ae(links, light_times, dt, *, doppler=True):
    _ensure_runtime()
    import cupy as cp

    order = ("12", "23", "31", "13", "32", "21")
    eta = {k: cp.ascontiguousarray(cp.asarray(links[j], dtype=cp.float64)) for j, k in enumerate(order)}
    ltt = {k: cp.ascontiguousarray(cp.asarray(light_times[j], dtype=cp.float64)) for j, k in enumerate(order)}

    def delay(path, values):
        total = cp.zeros_like(values)
        for i in range(len(path) - 1):
            total -= _shifted(ltt[path[i:i+2]], total / dt, 5)
        out = _shifted(values, total / dt, 31)
        if doppler:
            out *= 1 + cp.gradient(total, dt)
        return out

    xyz = []
    for i, j, k in ["123", "231", "312"]:
        p = delay(i+j, eta[j+i]) + eta[i+j]
        m = delay(i+k, eta[k+i]) + eta[i+k]
        p1 = delay(i+k+i, p) + m
        m1 = delay(i+j+i, m) + p
        p2 = delay(i+j+i+k+i, p1) + m1
        m2 = delay(i+k+i+j+i, m1) + p1
        xyz.append(p2 - m2)
    x, y, z = xyz
    return cp.stack((-(z-x)/cp.sqrt(2.), -(x-2*y+z)/cp.sqrt(6.)))


def _as_numpy(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def _validate_inputs(t, y_links, light_times, t_buffer):
    time = _as_numpy(t)
    if time.ndim != 1 or len(time) < 64 or np.iscomplexobj(time) or not np.all(np.isfinite(time)):
        raise ValueError("Invalid time grid")
    dt = float(time[1] - time[0])
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=0, atol=1e-9):
        raise ValueError("Sampling must be uniform and increasing")
    arrays = [_as_numpy(y_links), _as_numpy(light_times)]
    for value in arrays:
        if value.shape != (6, len(time)) or np.iscomplexobj(value) or not np.all(np.isfinite(value)):
            raise ValueError("Signals and light times must be finite real arrays of shape (6, N)")
    if np.any(arrays[1] <= 0):
        raise ValueError("Light times must be positive")
    if not np.isfinite(t_buffer) or t_buffer < 8*np.max(arrays[1]) + 64*dt:
        raise ValueError("Insufficient TDI edge buffer")
    trim = int(np.ceil(t_buffer / dt))
    if 2*trim >= len(time):
        raise ValueError("Edge buffer removes all samples")
    return time, dt, trim


def compute_tdi2_ae(t, y_links, light_times, *, t_buffer=2048., doppler=True):
    from .fastlisa import TDIResult

    time, dt, trim = _validate_inputs(t, y_links, light_times, t_buffer)
    channels = _compute_ae(y_links, light_times, dt, doppler=doppler)
    return TDIResult(
        t=time[trim:-trim],
        channels={c: channels[j, trim:-trim] for j, c in enumerate("AE")},
        projections=None,
        response_model=None,
        metadata={"backend": "recursive_cuda", "tdi": "2nd generation", "tdi_chan": "AE",
                  "input_kind": "precomputed_links",
                  "dt": dt, "t_buffer": float(t_buffer), "tdi_start_ind": trim,
                  "link_order": [12, 23, 31, 13, 32, 21], "delay_evaluation": "recursive",
                  "signal_interpolation_order": 31, "delay_interpolation_order": 5,
                  "doppler": bool(doppler)},
    )
