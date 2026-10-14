from dataclasses import dataclass

import numpy as np

from .recursive_tdi import _validate_inputs

LINK_ORDER = ("12", "23", "31", "13", "32", "21")


@dataclass
class NoisePacket:
    time_tcb_s: np.ndarray
    links_ffd: np.ndarray
    light_times_s: np.ndarray
    clock_correction_ae: np.ndarray | None = None
    link_order: tuple = LINK_ORDER

    def validate(self, trim_seconds):
        if tuple(self.link_order) != LINK_ORDER:
            raise ValueError("Link order must be 12,23,31,13,32,21; receiver first")
        time, dt, trim = _validate_inputs(self.time_tcb_s, self.links_ffd, self.light_times_s, trim_seconds)
        if self.clock_correction_ae is not None:
            value = np.asarray(self.clock_correction_ae)
            if value.shape != (2, len(time)) or np.iscomplexobj(value) or not np.all(np.isfinite(value)):
                raise ValueError("Clock correction must be a finite real array of shape (2, N)")
        return dt, trim


def packet_from_measurements(measurements, *, laser_frequency_hz, clock_correction):
    if not np.isfinite(laser_frequency_hz) or laser_frequency_hz <= 0:
        raise ValueError("Laser frequency must be positive and in Hz")
    required = {"sci_c", "ref_c", "tm_c", "ltt", "time", "fsample"}
    if clock_correction:
        required.update({"sci_sb", "ref_sb", "a", "b"})
    if not required.issubset(measurements):
        raise ValueError("Expected TCB-converted Triangle measurements with light times and required readouts")
    if not np.isfinite(measurements["fsample"]) or measurements["fsample"] <= 0:
        raise ValueError("Sampling frequency must be positive")
    time = np.asarray(measurements["time"]["1"])
    if any(not np.array_equal(time, np.asarray(measurements["time"][sc])) for sc in ["2", "3"]):
        raise ValueError("All spacecraft readouts must share the same TCB grid")
    from Triangle import TDI

    copied = {k: v.copy() if hasattr(v, "copy") else v for k, v in measurements.items()}
    engine = TDI(copied, copied["ltt"], fsample=copied["fsample"])
    engine.CalculateXi(doppler=True)
    engine.CalculateEta(doppler=True)
    correction = None
    if clock_correction:
        for c in "XYZ":
            path = "X2" if c == "X" else [getattr(engine, c+"2_plus"), getattr(engine, c+"2_minus")]
            engine.CalculateClockTDI(channel=path, channel_name=c+"2", doppler=True, modulation_correction=True)
        x, y, z = [np.asarray(engine.measurements[c+"2_q"]) / laser_frequency_hz for c in "XYZ"]
        correction = np.stack(((z-x)/np.sqrt(2.), (x-2*y+z)/np.sqrt(6.)))
    return NoisePacket(
        time_tcb_s=np.asarray(copied["time"]["1"]).copy(),
        links_ffd=-np.stack([np.asarray(engine.measurements["eta"][k]) for k in LINK_ORDER]) / laser_frequency_hz,
        light_times_s=np.stack([np.asarray(copied["ltt"][k]) for k in LINK_ORDER]),
        clock_correction_ae=correction,
    )


def compute_noise_ae(packet, *, backend, trim_seconds=2048.):
    dt, trim = packet.validate(trim_seconds)
    if backend == "triangle":
        from Triangle import TDI, MOSADict

        eta = MOSADict({k: -np.asarray(packet.links_ffd)[j] for j, k in enumerate(LINK_ORDER)})
        ltt = MOSADict({k: np.asarray(packet.light_times_s)[j] for j, k in enumerate(LINK_ORDER)})
        engine = TDI({"eta": eta}, ltt, fsample=1/dt)
        engine.FastMichelson(doppler=True, channel="AET")
        channels = np.stack([np.asarray(engine.measurements[c+"2"]) for c in "AE"])
    elif backend == "recursive_cuda":
        from .recursive_tdi import _compute_ae

        channels = _compute_ae(packet.links_ffd, packet.light_times_s, dt, doppler=True)
    else:
        raise ValueError("backend must be triangle or recursive_cuda")
    if packet.clock_correction_ae is not None:
        if backend == "recursive_cuda":
            import cupy as cp

            channels -= cp.asarray(packet.clock_correction_ae)
        else:
            channels -= packet.clock_correction_ae
    return np.asarray(packet.time_tcb_s)[trim:-trim], channels[:, trim:-trim]
