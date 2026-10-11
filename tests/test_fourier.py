from __future__ import annotations

import os
import unittest

import numpy as np

from gwdelta.fourier import fft, ifft, rfft, irfft


class FourierTests(unittest.TestCase):
    def test_complex_forward_matches_positive_exponential_sum(self):
        values = np.array([1+2j, -3j, 4., -2+1j, .5])
        k = np.arange(len(values))
        expected = np.exp(2j*np.pi*np.outer(k, k)/len(k))@values
        np.testing.assert_allclose(fft(values), expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(ifft(expected), values, rtol=1e-13, atol=1e-13)

    def test_real_forward_inverse_and_odd_lengths(self):
        for n in (15, 16):
            values = np.cos(np.arange(n)*.37)+np.sin(np.arange(n)*.21)
            np.testing.assert_allclose(rfft(values), fft(values)[:n//2+1], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(irfft(rfft(values), n=n), values, rtol=1e-13, atol=1e-13)
        with self.assertRaises(ValueError):
            rfft(np.array([1+1j]))

    def test_derivative_and_retarded_time_shift(self):
        n = 256
        dt = .125
        time = np.arange(n)*dt
        frequency = 7/(n*dt)
        omega = 2*np.pi*frequency
        phase = .31
        values = np.cos(omega*time+phase)
        derivative = -omega*np.sin(omega*time+phase)
        np.testing.assert_allclose(rfft(derivative)[7], -1j*omega*rfft(values)[7], rtol=1e-13)
        delay = .7
        delayed = np.cos(omega*(time-delay)+phase)
        np.testing.assert_allclose(rfft(delayed)[7], np.exp(1j*omega*delay)*rfft(values)[7], rtol=1e-13)

    def test_negative_frequency_reality_and_signal_invariance(self):
        values = np.sin(np.arange(127)*.27)+.4*np.cos(np.arange(127)*.19)
        spectrum = fft(values)
        np.testing.assert_allclose(spectrum[1:], np.conj(spectrum[:0:-1]), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(spectrum, np.conj(np.fft.fft(values)), rtol=1e-13, atol=1e-13)
        phase = np.exp(2j*np.pi*np.fft.fftfreq(len(values))*.7)
        migrated = ifft(spectrum*phase)
        legacy = np.fft.ifft(np.fft.fft(values)*np.conj(phase))
        np.testing.assert_allclose(migrated, legacy, rtol=1e-13, atol=1e-13)

    @unittest.skipUnless(os.environ.get("GWDELTA_RUN_CUDA_TESTS") == "1", "optional CUDA Fourier test")
    def test_cupy_forward_inverse(self):
        from gwdelta import ensure_cuda_dll_directories
        ensure_cuda_dll_directories()
        import cupy as cp
        values = np.sin(np.arange(257)*.37)
        gpu = rfft(cp.asarray(values))
        self.assertIsInstance(gpu, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(gpu), rfft(values), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cp.asnumpy(irfft(gpu, n=len(values))), values, rtol=1e-12, atol=1e-12)
        complex_values = values+1j*np.cos(np.arange(257)*.19)
        np.testing.assert_allclose(cp.asnumpy(fft(cp.asarray(complex_values))), fft(complex_values), rtol=1e-12, atol=1e-12)
