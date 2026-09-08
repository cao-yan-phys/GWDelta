from __future__ import annotations

import unittest

import numpy as np

from gwdelta import (
    FastLISAResponseTDI,
    make_lisa_simple_orbits,
    compute_tdi2_ae,
)


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

    def test_precise_tdi2_preserves_common_constant_links(self) -> None:
        dt = 2.0
        times = np.arange(4096, dtype=float) * dt
        orbits = make_lisa_simple_orbits(
            duration=9000.0,
            orbit_dt=60.0,
            force_backend="cpu",
        )
        result = compute_tdi2_ae(
            times,
            np.full((6, len(times)), 1.0e-20),
            orbits=orbits,
            t_buffer=1500.0,
        )
        arrays = result.as_numpy()
        np.testing.assert_allclose(arrays["A"], 0.0, rtol=0.0, atol=1.0e-34)
        np.testing.assert_allclose(arrays["E"], 0.0, rtol=0.0, atol=1.0e-34)
        self.assertEqual(result.metadata["backend"], "scipy_cubic_precomputed_tdi2")
        self.assertEqual(result.metadata["delay_evaluation"], "recursive")

    def test_precise_tdi2_agrees_with_fast_backend_for_smooth_links(self) -> None:
        dt = 2.0
        times = np.arange(4096, dtype=float) * dt
        orbits = make_lisa_simple_orbits(
            duration=9000.0,
            orbit_dt=60.0,
            force_backend="cpu",
        )
        phase = 2.0 * np.pi * 0.003 * times
        links = np.vstack(
            [
                (index + 1.0) * 1.0e-21 * np.sin(phase + 0.2 * index)
                for index in range(6)
            ]
        )
        fast = FastLISAResponseTDI(
            orbits=orbits,
            order=15,
            tdi="2nd generation",
            tdi_chan="AE",
            force_backend="cpu",
            t_buffer=1500.0,
            trim_garbage=True,
        ).compute_links(times, links).as_numpy()
        precise = compute_tdi2_ae(
            times,
            links,
            orbits=orbits,
            t_buffer=1500.0,
        ).as_numpy()
        np.testing.assert_array_equal(precise["t"], fast["t"])
        np.testing.assert_allclose(precise["A"], fast["A"], rtol=2.0e-4, atol=1.0e-30)
        np.testing.assert_allclose(precise["E"], fast["E"], rtol=2.0e-4, atol=1.0e-30)

    def test_time_domain_polarizations_feed_tdi(self) -> None:
        dt = 2.0
        times = np.arange(4096, dtype=float) * dt
        source_time = np.arange(-1200.0, 9400.0, dt)
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
        phase = 2.0 * np.pi * 0.003 * source_time
        result = response.compute_polarizations(
            times,
            {
                "plus": 1.0e-21 * np.cos(phase),
                "vector_x": 0.2e-21 * np.sin(phase),
            },
            source_time_s=source_time,
            lam=0.3,
            beta=0.4,
            quadrature_order=8,
        )
        arrays = result.as_numpy()
        self.assertEqual(set(arrays) - {"t", "projections"}, {"A", "E"})
        self.assertTrue(np.all(np.isfinite(arrays["A"])))
        self.assertTrue(np.all(np.isfinite(arrays["E"])))
        self.assertEqual(result.metadata["input_kind"], "sampled_plane_wave_polarizations")
        self.assertEqual(result.metadata["polarizations"], ["plus", "vector_x"])


if __name__ == "__main__":
    unittest.main()
