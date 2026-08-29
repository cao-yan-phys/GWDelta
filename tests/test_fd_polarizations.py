from __future__ import annotations

import unittest

import numpy as np

from gwdelta import (
    POLARIZATION_NAMES,
    StaticTaijiFDResponse,
    link_fd_polarization_response,
    link_fd_response,
    polarization_tensors,
    sky_basis,
    static_taiji_positions,
)


class StaticPolarizationResponseTests(unittest.TestCase):
    def test_six_tensors_are_orthonormal_and_have_expected_propagation_parts(
        self,
    ) -> None:
        lam = 0.3
        beta = -0.4
        k, u, v = sky_basis(lam, beta)
        tensors = polarization_tensors(lam, beta)
        self.assertEqual(tuple(tensors), POLARIZATION_NAMES)
        tensor_stack = np.stack([tensors[name] for name in POLARIZATION_NAMES])
        gram = np.einsum("aij,bij->ab", tensor_stack, tensor_stack)
        np.testing.assert_allclose(
            gram, 2.0 * np.eye(len(POLARIZATION_NAMES)), atol=1.0e-15
        )
        contractions = np.einsum("i,aij->aj", k, tensor_stack)
        np.testing.assert_allclose(contractions[0], 0.0, atol=1.0e-15)
        np.testing.assert_allclose(contractions[1], 0.0, atol=1.0e-15)
        np.testing.assert_allclose(contractions[2], u)
        np.testing.assert_allclose(contractions[3], v)
        np.testing.assert_allclose(contractions[4], 0.0, atol=1.0e-15)
        np.testing.assert_allclose(contractions[5], np.sqrt(2.0) * k)

    def test_gr_transfers_are_unchanged(self) -> None:
        frequency = np.asarray([1.0e-4, 1.2e-3, 4.0e-3])
        kwargs = {
            "lam": 1.1,
            "beta": -0.2,
            "positions_m": static_taiji_positions(),
        }
        plus, cross = link_fd_response(frequency, **kwargs)
        responses = link_fd_polarization_response(frequency, **kwargs)
        np.testing.assert_allclose(responses["plus"], plus, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(responses["cross"], cross, rtol=0.0, atol=0.0)
        self.assertEqual(set(responses), set(POLARIZATION_NAMES))
        self.assertTrue(
            all(response.shape == (6, len(frequency)) for response in responses.values())
        )

    def test_static_tdi_accepts_mixed_polarizations(self) -> None:
        frequency = np.linspace(5.0e-4, 3.0e-3, 11)
        h_plus = 1.0e-22 * np.exp(0.3j * frequency / frequency[-1])
        h_cross = -0.4j * h_plus
        h_breathing = 0.2 * h_plus
        response = StaticTaijiFDResponse()

        expected = response.ae(
            frequency,
            h_plus,
            h_cross,
            lam=0.2,
            beta=0.5,
            tdi="2nd generation",
        )
        actual = response.ae_polarizations(
            frequency,
            {"plus": h_plus, "cross": h_cross},
            lam=0.2,
            beta=0.5,
            tdi="2nd generation",
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=1.0e-40)

        mixed = response.ae_polarizations(
            frequency,
            {"plus": h_plus, "cross": h_cross, "breathing": h_breathing},
            lam=0.2,
            beta=0.5,
            tdi="2nd generation",
        )
        self.assertTrue(all(np.all(np.isfinite(values)) for values in mixed))
        self.assertGreater(np.linalg.norm(mixed[0] - actual[0]), 0.0)

    def test_static_tdi_rejects_invalid_polarization_inputs(self) -> None:
        frequency = np.asarray([1.0e-3, 2.0e-3])
        response = StaticTaijiFDResponse()
        with self.assertRaisesRegex(ValueError, "nonempty polarization mapping"):
            response.ae_polarizations(
                frequency, {}, lam=0.0, beta=0.0
            )
        with self.assertRaisesRegex(ValueError, "unknown polarization names"):
            response.ae_polarizations(
                frequency, {"tensor": np.ones(2)}, lam=0.0, beta=0.0
            )
        with self.assertRaisesRegex(ValueError, "plus spectrum must have shape"):
            response.ae_polarizations(
                frequency, {"plus": np.ones(3)}, lam=0.0, beta=0.0
            )


if __name__ == "__main__":
    unittest.main()
