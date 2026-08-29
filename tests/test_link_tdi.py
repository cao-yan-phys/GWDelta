from __future__ import annotations

import unittest

import numpy as np

from gwdelta import FastLISAResponseTDI, make_lisa_simple_orbits


class LinkTDITests(unittest.TestCase):
    def test_precomputed_links_reproduce_standard_response(self) -> None:
        dt = 2.0
        times = np.arange(4096, dtype=float) * dt
        orbits = make_lisa_simple_orbits(
            duration=9000.0,
            orbit_dt=60.0,
            force_backend="cpu",
        )
        response = FastLISAResponseTDI(
            orbits=orbits,
            order=5,
            tdi="2nd generation",
            tdi_chan="AE",
            force_backend="cpu",
            t_buffer=1500.0,
            trim_garbage=True,
        )
        phase = 2.0 * np.pi * 0.003 * times
        standard = response.compute(
            times,
            1.0e-21 * np.cos(phase),
            0.7e-21 * np.sin(phase),
            lam=0.3,
            beta=0.4,
        )
        links = standard.as_numpy()["projections"]
        direct = response.compute_links(times, links)
        standard_arrays = standard.as_numpy()
        direct_arrays = direct.as_numpy()

        np.testing.assert_array_equal(direct_arrays["t"], standard_arrays["t"])
        np.testing.assert_allclose(
            direct_arrays["A"], standard_arrays["A"], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            direct_arrays["E"], standard_arrays["E"], rtol=0.0, atol=0.0
        )
        self.assertEqual(direct.metadata["input_kind"], "precomputed_links")
        self.assertEqual(direct.link_order, [12, 23, 31, 13, 32, 21])

    def test_precomputed_link_shape_is_strict(self) -> None:
        times = np.arange(512, dtype=float)
        orbits = make_lisa_simple_orbits(
            duration=600.0, orbit_dt=30.0, force_backend="cpu"
        )
        response = FastLISAResponseTDI(
            orbits=orbits,
            order=5,
            force_backend="cpu",
            t_buffer=200.0,
        )
        with self.assertRaisesRegex(ValueError, "y_links must have shape"):
            response.compute_links(times, np.zeros((5, len(times))))


if __name__ == "__main__":
    unittest.main()
