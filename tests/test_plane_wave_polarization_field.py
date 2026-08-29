from __future__ import annotations

import unittest

import numpy as np

from gwdelta import (
    C_SI,
    PlaneWavePolarizationField,
    SampledOrbits,
    WeakFieldLinkResponse,
    link_fd_polarization_response,
)


def static_orbits(*, arm_m: float = 2.5e9, duration_s: float = 40000.0):
    center = np.asarray([1.495978707e11, -2.0e10, 1.0e10])
    radius = arm_m / np.sqrt(3.0)
    angles = np.deg2rad([0.0, 120.0, 240.0])
    positions = center + np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(3)]
    )
    times = np.asarray([0.0, duration_s])
    return (
        SampledOrbits(
            times,
            np.repeat(positions[np.newaxis, :, :], len(times), axis=0),
            armlength=arm_m,
            force_backend="cpu",
        ),
        positions,
    )


def dynamic_orbits(*, arm_m: float = 2.5e9, duration_s: float = 40000.0):
    static, positions = static_orbits(arm_m=arm_m, duration_s=duration_s)
    del static
    center = np.mean(positions, axis=0)
    relative = positions - center
    time = np.arange(0.0, duration_s + 1.0, 100.0)
    angle = 2.0e-5 * time
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated = np.empty((len(time), 3, 3))
    rotated[..., 0] = (
        cos_angle[:, np.newaxis] * relative[np.newaxis, :, 0]
        - sin_angle[:, np.newaxis] * relative[np.newaxis, :, 1]
    )
    rotated[..., 1] = (
        sin_angle[:, np.newaxis] * relative[np.newaxis, :, 0]
        + cos_angle[:, np.newaxis] * relative[np.newaxis, :, 1]
    )
    rotated[..., 2] = relative[np.newaxis, :, 2]
    drift = np.column_stack([500.0 * time, -300.0 * time, np.zeros_like(time)])
    return SampledOrbits(
        time,
        center[np.newaxis, np.newaxis, :] + drift[:, np.newaxis, :] + rotated,
        armlength=arm_m,
        force_backend="cpu",
    )


class PlaneWavePolarizationFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orbits, cls.positions = static_orbits()
        cls.frequency_hz = 3.0e-4
        cls.angular_frequency = 2.0 * np.pi * cls.frequency_hz
        cls.lam = 0.4
        cls.beta = -0.3
        cls.amplitude = 1.0e-20
        cls.source_time = np.arange(-2000.0, 42001.0, 2.0)
        cls.response_time = np.arange(1000.0, 39001.0, 100.0)

    def _field(self, name: str) -> PlaneWavePolarizationField:
        samples = self.amplitude * np.cos(self.angular_frequency * self.source_time)
        return PlaneWavePolarizationField(
            self.source_time,
            {name: samples},
            lam=self.lam,
            beta=self.beta,
        )

    def test_each_time_domain_polarization_matches_static_frequency_response(self) -> None:
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=32, force_backend="cpu"
        )
        expected = link_fd_polarization_response(
            np.asarray([self.frequency_hz]),
            lam=self.lam,
            beta=self.beta,
            positions_m=self.positions,
        )
        phase = np.exp(1j * self.angular_frequency * self.response_time)
        for name, response in expected.items():
            actual = engine.compute(self.response_time, self._field(name)).as_numpy()[
                "direct"
            ]
            reference = self.amplitude * np.real(response[:, :1] * phase[np.newaxis, :])
            np.testing.assert_allclose(actual, reference, rtol=3.0e-5, atol=2.0e-26)

    def test_dynamic_orbit_time_domain_response_is_finite(self) -> None:
        orbits = dynamic_orbits()
        engine = WeakFieldLinkResponse(
            orbits=orbits, quadrature_order=16, force_backend="cpu"
        )
        result = engine.compute(self.response_time, self._field("vector_x")).as_numpy()
        self.assertEqual(result["direct"].shape, (6, len(self.response_time)))
        self.assertTrue(np.all(np.isfinite(result["direct"])))
        self.assertGreater(np.linalg.norm(result["direct"]), 0.0)

    def test_retarded_time_support_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not cover the requested retarded times"):
            PlaneWavePolarizationField(
                np.asarray([0.0, 1.0]),
                {"plus": np.asarray([0.0, 0.0])},
                lam=0.0,
                beta=0.0,
            ).metric(
                np.asarray([0.0]),
                np.asarray([[10.0 * C_SI, 0.0, 0.0]]),
            )


if __name__ == "__main__":
    unittest.main()
