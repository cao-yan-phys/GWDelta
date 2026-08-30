from __future__ import annotations

import unittest

import numpy as np

from gwdelta import (
    C_SI,
    equal_arm_aet_noise_psd,
    frozen_tdi2_detector_noise_covariance,
    frozen_tdi2_light_times_from_positions,
    frozen_tdi2_noise_covariance,
    frozen_tdi2_t_low_frequency_leakage,
    one_way_noise_psd,
)


class FrozenTDI2NoiseTests(unittest.TestCase):
    @staticmethod
    def _equal_light_times(arm_m: float = 2.5e9) -> dict[tuple[int, int], float]:
        light_time = arm_m / C_SI
        return {
            (1, 2): light_time,
            (2, 1): light_time,
            (1, 3): light_time,
            (3, 1): light_time,
            (2, 3): light_time,
            (3, 2): light_time,
        }

    @staticmethod
    def _unequal_light_times() -> dict[tuple[int, int], float]:
        return {
            (1, 2): 8.30,
            (2, 1): 8.30,
            (1, 3): 8.35,
            (3, 1): 8.35,
            (2, 3): 8.25,
            (3, 2): 8.25,
        }

    def test_primitive_and_link_contractions_agree_and_are_physical(self) -> None:
        frequency = np.geomspace(1.0e-4, 4.0e-2, 17)
        primitive = frozen_tdi2_noise_covariance(
            frequency,
            self._unequal_light_times(),
            oms_psd=1.0,
            tm_psd=0.5,
            basis="AET",
        )
        links = frozen_tdi2_noise_covariance(
            frequency,
            self._unequal_light_times(),
            oms_psd=1.0,
            tm_psd=0.5,
            basis="AET",
            method="links",
        )
        np.testing.assert_allclose(primitive, links, rtol=3.0e-12, atol=1.0e-25)
        np.testing.assert_allclose(
            primitive,
            np.swapaxes(primitive.conj(), 1, 2),
            rtol=3.0e-12,
            atol=1.0e-25,
        )
        eigenvalues = np.linalg.eigvalsh(primitive)
        self.assertGreaterEqual(float(eigenvalues.min()), -1.0e-12)

    def test_equal_arm_diagonals_recover_existing_aet_psds(self) -> None:
        frequency = np.asarray([2.0e-4, 7.0e-4, 2.0e-3, 8.0e-3, 2.0e-2])
        covariance = frozen_tdi2_detector_noise_covariance(
            frequency,
            self._equal_light_times(),
            "lisa",
            basis="AET",
        )
        expected = equal_arm_aet_noise_psd(
            frequency,
            "lisa",
            channels="AET",
            tdi_generation="second",
        )
        actual = np.diagonal(covariance, axis1=1, axis2=2).T.real
        np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-70)
        off_diagonal = covariance[:, (0, 0, 1), (1, 2, 2)]
        self.assertLess(float(np.max(np.abs(off_diagonal / actual.T))), 3.0e-12)

    def test_equal_arm_xyz_closed_forms(self) -> None:
        arm_m = 2.5e9
        frequency = np.asarray([2.0e-4, 7.0e-4, 2.0e-3, 8.0e-3, 2.0e-2])
        covariance = frozen_tdi2_detector_noise_covariance(
            frequency,
            self._equal_light_times(arm_m),
            "lisa",
            basis="XYZ",
        )
        tm_psd, oms_psd = one_way_noise_psd(frequency, "lisa")
        x = 2.0 * np.pi * frequency * arm_m / C_SI
        x2_psd = (
            64.0
            * np.sin(x) ** 2
            * np.sin(2.0 * x) ** 2
            * (oms_psd + (3.0 + np.cos(2.0 * x)) * tm_psd)
        )
        xy2_csd = (
            -16.0
            * np.sin(2.0 * x)
            * np.sin(x)
            * np.sin(2.0 * x) ** 2
            * (oms_psd + 4.0 * tm_psd)
        )
        expected_diagonal = np.broadcast_to(x2_psd[:, np.newaxis], (len(frequency), 3))
        np.testing.assert_allclose(
            np.diagonal(covariance, axis1=1, axis2=2).real,
            expected_diagonal,
            rtol=1.0e-10,
            atol=1.0e-70,
        )
        np.testing.assert_allclose(
            covariance[:, 0, 1].real,
            xy2_csd,
            rtol=1.0e-10,
            atol=1.0e-70,
        )
        relative_imaginary = np.max(
            np.abs(covariance[:, 0, 1].imag)
            / np.maximum(np.abs(xy2_csd), np.finfo(float).tiny)
        )
        self.assertLess(float(relative_imaginary), 3.0e-12)

    def test_low_frequency_unequal_arm_t_leakage(self) -> None:
        frequency = np.asarray([1.0e-6, 3.0e-6, 1.0e-5])
        full = frozen_tdi2_noise_covariance(
            frequency,
            self._unequal_light_times(),
            oms_psd=1.0,
            tm_psd=0.5,
            basis="AET",
        )[:, 2, 2].real
        leading = frozen_tdi2_t_low_frequency_leakage(
            frequency,
            self._unequal_light_times(),
            oms_psd=1.0,
            tm_psd=0.5,
        )
        np.testing.assert_allclose(full, leading, rtol=6.0e-4, atol=1.0e-25)

    def test_directed_delays_and_noise_psds_are_supported(self) -> None:
        frequency = np.asarray([4.0e-4, 2.0e-3])
        delays = {
            12: 8.30,
            21: 8.31,
            13: 8.35,
            31: 8.34,
            23: 8.25,
            32: 8.26,
        }
        oms = {link: 1.0 + 0.1 * index for index, link in enumerate(delays)}
        tm = {link: np.asarray([0.4 + 0.01 * index, 0.5 + 0.01 * index]) for index, link in enumerate(delays)}
        covariance = frozen_tdi2_noise_covariance(
            frequency,
            delays,
            oms_psd=oms,
            tm_psd=tm,
            basis="XYZ",
        )
        self.assertTrue(np.all(np.isfinite(covariance)))
        np.testing.assert_allclose(
            covariance,
            np.swapaxes(covariance.conj(), 1, 2),
            rtol=3.0e-12,
            atol=1.0e-25,
        )

    def test_wolfram_stable_t2_reference(self) -> None:
        """Reference values from TDI_T2_stable_check.wl on a fixed triangle."""

        positions = np.asarray(
            [[0.0, 0.0, 0.0], [2.5e9, 0.0, 0.0], [0.8e9, 2.3e9, 0.0]]
        )
        frequency = np.asarray([1.0e-5, 3.0e-4, 3.0e-3, 3.0e-2])
        expected_t = np.asarray(
            [
                6.387718868901819e-47,
                1.0533176549562108e-46,
                9.326688883007658e-45,
                1.8323315526821634e-40,
            ]
        )
        covariance = frozen_tdi2_detector_noise_covariance(
            frequency,
            frozen_tdi2_light_times_from_positions(positions),
            "lisa",
            basis="AET",
        )
        np.testing.assert_allclose(covariance[:, 2, 2].real, expected_t, rtol=5.0e-10)


if __name__ == "__main__":
    unittest.main()
