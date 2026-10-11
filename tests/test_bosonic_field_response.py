from __future__ import annotations

import unittest
import os

import numpy as np
from scipy.integrate import quad_vec, trapezoid
from scipy.interpolate import CubicSpline

from gwdelta import SampledOrbits
from gwdelta.bosonic_field_response import BosonicFieldLinkResponse, ScalarPlaneWave, VectorPlaneWave, static_link_amplitudes
from gwdelta.fd_response import C_SI, DEFAULT_LINKS, second_generation_xyz_from_links, aet_from_xyz
from gwdelta.fourier import rfft
from gwdelta.weak_field import build_link_geometry


POSITIONS = np.array([[0., 0., 0.], [2.5e9, 0., 0.], [.7e9, 2.1e9, .3e9]])
K = np.array([.3, -.4, np.sqrt(.75)])


def orbits(velocity=None, positions=POSITIONS, duration=3000.):
    t = np.array([0., duration])
    v = np.zeros((3, 3)) if velocity is None else np.broadcast_to(velocity, (3, 3))
    x = positions[None]+t[:, None, None]*v
    return SampledOrbits(t, x, armlength=2.5e9, force_backend="cpu")


def initial_steady(field, positions=POSITIONS):
    f = field.frequencies_hz
    vg = np.sqrt(1-(field.mass_frequency_hz/f)**2)
    phase = np.exp(2j*np.pi*f*((positions-field.reference_position_m)@field.direction)[:, None]*vg/C_SI
                   +2j*np.pi*f*field.reference_time_s)
    return C_SI*np.real(phase@field.velocity_coefficients())


class BosonicFieldResponseTests(unittest.TestCase):
    def test_static_scalar_and_vector_match_endpoint_formula(self):
        for field in (ScalarPlaneWave(.004, 1e-15*(1+.4j), K, .002),
                      VectorPlaneWave(.004, [1e-15, 2e-15j, -3e-15], K, .002)):
            amplitudes = static_link_amplitudes(field, POSITIONS)
            geometry = build_link_geometry(orbits(), np.array([777.]))
            vg = np.sqrt(1-(field.mass_frequency_hz/field.frequencies_hz)**2)
            for link in range(6):
                xe, xr = geometry.x_emission_m[link, 0], geometry.x_reception_m[link, 0]
                delay = geometry.light_time_s[link, 0]
                receiver = np.exp(2j*np.pi*field.frequencies_hz*vg*np.dot(K, xr)/C_SI)
                emitter = np.exp(2j*np.pi*field.frequencies_hz*(delay+vg*np.dot(K, xe)/C_SI))
                expected = -geometry.direction[link, 0]@field.velocity_coefficients().T*(receiver-emitter)
                np.testing.assert_allclose(amplitudes[link], expected, rtol=2e-13, atol=1e-29)

    def test_static_time_domain_matches_frequency_amplitudes_and_fft(self):
        equal = np.array([[0., 0., 0.], [2.5e9, 0., 0.], [1.25e9, np.sqrt(3)*1.25e9, 0.]])
        t = 200.+np.arange(1024.)
        for positions in (POSITIONS, equal):
            for field in (ScalarPlaneWave(5/1024, 1e-15*(1+.2j), K, .002),
                          VectorPlaneWave(5/1024, [1e-15, 2e-15j, 3e-15], K, .002)):
                response = BosonicFieldLinkResponse(orbits(positions=positions), chunk_size=37)
                result = response.compute(t, field, initial_velocity_m_s=initial_steady(field, positions), integration_step_s=73.)
                amplitude = static_link_amplitudes(field, positions)[:, 0]
                expected = np.real(amplitude[:, None]*np.exp(-2j*np.pi*field.frequencies_hz[0]*t))
                error = np.linalg.norm(result.total-expected)/np.linalg.norm(expected)
                self.assertLess(error, 2e-12)
                fft = rfft(result.total, axis=1)[:, 5]*2/len(t)*np.exp(2j*np.pi*field.frequencies_hz[0]*t[0])
                np.testing.assert_allclose(fft, amplitude, rtol=2e-12, atol=1e-28)

    def test_uniform_motion_matches_independent_force_integral(self):
        velocities = np.array([[30000., 10000., 0.], [-10000., 20000., 5000.], [5000., -30000., -4000.]])
        t = np.array([230., 711., 1350.])
        orbit = orbits(velocities)
        geometry = build_link_geometry(orbit, t)
        for field in (ScalarPlaneWave(.003, 1e-15*(1+.3j), K, .001),
                      VectorPlaneWave(.003, [2e-15, 1e-15j, -1e-15], K, .001)):
            omega = 2*np.pi*field.frequencies_hz[0]
            vg = np.sqrt(1-(field.mass_frequency_hz/.003)**2)
            if isinstance(field, ScalarPlaneWave):
                def force(time, sc):
                    position = POSITIONS[sc]+velocities[sc]*time
                    phi = field.amplitudes[0]*np.exp(-1j*omega*(time-vg*np.dot(K, position)/C_SI))
                    return np.real(1j*omega*C_SI*vg*phi*K)
            else:
                def force(time, sc):
                    position = POSITIONS[sc]+velocities[sc]*time
                    phase = np.exp(-1j*omega*(time-vg*np.dot(K, position)/C_SI))
                    return np.real(-1j*omega*C_SI*(vg*field.temporal_amplitudes[0]*K-field.amplitudes[0])*phase)
            result = BosonicFieldLinkResponse(orbit).compute(t, field, integration_step_s=97.)
            expected = np.empty_like(result.total)
            for link, code in enumerate(DEFAULT_LINKS):
                receiver, emitter = code//10-1, code%10-1
                for j, tr in enumerate(t):
                    vr = quad_vec(lambda time: force(time, receiver), 0., tr, epsabs=1e-22, epsrel=1e-12)[0]
                    ve = quad_vec(lambda time: force(time, emitter), 0., geometry.t_emission_s[link, j], epsabs=1e-22, epsrel=1e-12)[0]
                    expected[link, j] = -np.dot(geometry.direction[link, j], vr-ve)/C_SI
            self.assertLess(np.linalg.norm(result.total-expected)/np.linalg.norm(expected), 2e-11)

    def test_scalar_limits_and_stable_small_frequency(self):
        along_x = np.array([1., 0., 0.])
        perpendicular = ScalarPlaneWave(.01, 1., [0., 1., 0.])
        self.assertEqual(static_link_amplitudes(perpendicular, POSITIONS)[5, 0], 0.)
        massless = ScalarPlaneWave(.01, 1., along_x)
        self.assertEqual(static_link_amplitudes(massless, POSITIONS)[5, 0], 0.)
        massive = ScalarPlaneWave(.01, 1., along_x, .005)
        self.assertGreater(abs(static_link_amplitudes(massive, POSITIONS)[5, 0]), 0.)
        low = ScalarPlaneWave(1e-18, 1., [-1., 0., 0.])
        expected = 2j*np.pi*1e-18*2*2.5e9/C_SI
        np.testing.assert_allclose(static_link_amplitudes(low, POSITIONS)[5, 0], expected, rtol=1e-12)

    def test_vector_lorenz_longitudinal_and_pure_gauge(self):
        pure_gauge = VectorPlaneWave(.005, K*(2+1j), K)
        np.testing.assert_allclose(static_link_amplitudes(pure_gauge, POSITIONS), 0., atol=2e-15)
        massive = VectorPlaneWave(.005, K*(2+1j), K, .003)
        np.testing.assert_allclose(massive.velocity_coefficients(), -.36*(2+1j)*K[None], rtol=1e-14)
        with self.assertRaisesRegex(ValueError, "Lorenz"):
            VectorPlaneWave(.005, K, K, temporal_amplitudes=0.)
        threshold = ScalarPlaneWave(.005, 1., K, .005)
        np.testing.assert_array_equal(threshold.velocity_coefficients(), np.zeros((1, 3)))

    def test_spectrum_quadrature_and_superposition(self):
        frequencies = np.array([.001, .002, .005, .01])
        spectrum = np.array([1., 2j, -1., .5])
        field = ScalarPlaneWave.from_spectrum(frequencies, spectrum, K)
        expected = -2*trapezoid(spectrum, frequencies)*K
        np.testing.assert_allclose(np.sum(field.velocity_coefficients(), axis=0), expected)
        combined = static_link_amplitudes(field, POSITIONS).sum(axis=1)
        separate = sum(static_link_amplitudes(ScalarPlaneWave(f, a, K), POSITIONS)[:, 0]
                       for f, a in zip(field.frequencies_hz, field.amplitudes))
        np.testing.assert_allclose(combined, separate, rtol=1e-14)

    def test_parameters_initial_values_and_no_input_mutation(self):
        position = np.zeros(3)
        ScalarPlaneWave(.001, 1., K, reference_position_m=position)
        self.assertTrue(position.flags.writeable)
        for frequency, mass in ((0., 0.), (-1., 0.), (.001, .002), (.001, -1.)):
            with self.assertRaises(ValueError):
                ScalarPlaneWave(frequency, 1., K, mass)
        with self.assertRaises(ValueError):
            ScalarPlaneWave(.001, 1., [0., 0., 0.])
        with self.assertRaises(ValueError):
            ScalarPlaneWave.from_spectrum([.001, .003], [1., 0.], K, mass_frequency_hz=.002)
        field = ScalarPlaneWave(.001, 0., K)
        response = BosonicFieldLinkResponse(orbits())
        with self.assertRaisesRegex(ValueError, "initial time"):
            response.compute([0., 100.], field)
        with self.assertRaises(ValueError):
            response.compute([100., 200.], field, integration_step_s=0.)
        initial = np.arange(9).reshape(3, 3)*1e-6
        result = response.compute([100., 200.], field, initial_velocity_m_s=initial)
        for link, code in enumerate(DEFAULT_LINKS):
            expected = -np.sum(result.geometry.direction[link]*(initial[code//10-1]-initial[code%10-1]), axis=-1)/C_SI
            np.testing.assert_allclose(result.total[link], expected, atol=1e-28)

    def test_static_tdi2_matches_explicit_retarded_series(self):
        from gwdelta.fd_response import SECOND_GEN_X_COMBINATIONS, cyclic_permutation
        field = ScalarPlaneWave(.003, 1e-15*(1+.5j), K, .001)
        links = static_link_amplitudes(field, POSITIONS)
        xyz = second_generation_xyz_from_links(field.frequencies_hz, links, positions_m=POSITIONS)
        reference = []
        t = 321.
        phase = np.exp(-2j*np.pi*.003*t)
        for permutation in range(3):
            result = 0.
            for term in SECOND_GEN_X_COMBINATIONS:
                code = cyclic_permutation(term['link'], permutation)
                delay = sum(np.linalg.norm(POSITIONS[cyclic_permutation(d, permutation)//10-1]-POSITIONS[cyclic_permutation(d, permutation)%10-1])/C_SI for d in term['links_for_delay'])
                result += term['sign']*np.real(links[DEFAULT_LINKS.index(code), 0]*np.exp(-2j*np.pi*.003*(t-delay)))
            reference.append(result)
        np.testing.assert_allclose(np.real(np.array(xyz)[:, 0]*phase), reference, rtol=1e-12, atol=1e-28)
        self.assertTrue(np.all(np.isfinite(aet_from_xyz(*xyz))))

    def test_curved_trajectory_converges_to_force_integral(self):
        support = np.arange(0., 1601., 10.)
        omega_orbit = .002
        offset = 5e8*np.column_stack((np.cos(omega_orbit*support), np.sin(omega_orbit*support), np.zeros(len(support))))
        positions = POSITIONS[None]+offset[:, None]
        orbit = SampledOrbits(support, positions, armlength=2.5e9, force_backend="cpu")
        trajectory = CubicSpline(support, positions, axis=0)
        time = np.array([230., 777., 1450.])
        geometry = build_link_geometry(orbit, time)
        field = ScalarPlaneWave(.003, 1e-15*(1+.2j), K, .001)
        omega = 2*np.pi*.003
        vg = np.sqrt(1-(.001/.003)**2)
        def force(t, sc):
            phase = np.exp(-1j*omega*(t-vg*np.dot(K, trajectory(t)[sc])/C_SI))
            return np.real(1j*omega*C_SI*vg*field.amplitudes[0]*phase*K)
        expected = np.empty((6, len(time)))
        for link, code in enumerate(DEFAULT_LINKS):
            receiver, emitter = code//10-1, code%10-1
            for j, tr in enumerate(time):
                vr = quad_vec(lambda t: force(t, receiver), 0., tr, epsabs=1e-22, epsrel=1e-11)[0]
                ve = quad_vec(lambda t: force(t, emitter), 0., geometry.t_emission_s[link, j], epsabs=1e-22, epsrel=1e-11)[0]
                expected[link, j] = -np.dot(geometry.direction[link, j], vr-ve)/C_SI
        errors = []
        for step in (20., 10., 5.):
            result = BosonicFieldLinkResponse(orbit).compute(time, field, integration_step_s=step)
            errors.append(np.linalg.norm(result.total-expected)/np.linalg.norm(expected))
        self.assertLess(errors[-1], 2e-6)
        self.assertLess(errors[1], errors[0]/3)
        self.assertLess(errors[2], errors[1]/3)

    def test_frequency_and_time_chunks_do_not_change_results(self):
        f = np.array([.001, .002, .003, .004])
        field = VectorPlaneWave(f, np.array([[1., 2j, 3.], [2j, -1., 1j], [3., -2j, 1.], [1j, 2., -3.]])*1e-15, K, .0005)
        time = np.arange(100., 1001., 10.)
        orbit = orbits([30000., -10000., 0.])
        a = BosonicFieldLinkResponse(orbit, chunk_size=13, frequency_chunk_size=1).compute(time, field, integration_step_s=17.)
        b = BosonicFieldLinkResponse(orbit, chunk_size=1000, frequency_chunk_size=64).compute(time, field, integration_step_s=17.)
        self.assertLess(np.linalg.norm(a.total-b.total)/np.linalg.norm(b.total), 1e-12)

    def test_vector_spectrum_and_reference_coordinates(self):
        f = np.array([.001, .002, .004])
        spatial = np.array([[1., 2j, 3.], [2., 3j, 4.], [3., 4j, 5.]])*1e-15
        field = VectorPlaneWave.from_spectrum(f, spatial, K, mass_frequency_hz=.0005)
        vg = np.sqrt(1-(.0005/f)**2)
        np.testing.assert_allclose(field.temporal_amplitudes, vg*(field.amplitudes@K))
        shift = np.array([1e10, -2e10, 3e10])
        shifted = VectorPlaneWave(field.frequencies_hz, field.amplitudes, K, .0005,
                                  reference_time_s=71., reference_position_m=shift)
        a = BosonicFieldLinkResponse(orbits()).compute([200., 700.], field)
        base = SampledOrbits(np.array([71., 3071.]), np.broadcast_to(POSITIONS+shift, (2, 3, 3)).copy(),
                             armlength=2.5e9, force_backend="cpu")
        b = BosonicFieldLinkResponse(base).compute([271., 771.], shifted)
        self.assertLess(np.linalg.norm(a.total-b.total)/np.linalg.norm(a.total), 1e-11)

    @unittest.skipUnless(os.environ.get("GWDELTA_RUN_CUDA_TESTS") == "1", "optional CUDA response test")
    def test_realistic_lisa_cuda_links_and_tdi(self):
        from gwdelta import make_orbits_from_spec, FastLISAResponseTDI
        from gwdelta.precise_tdi import compute_tdi2_ae
        import cupy as cp
        orbit = make_orbits_from_spec({"base": "esa", "orbit_dt": 300., "use_project_phase_defaults": False},
                                     duration=12000., force_backend="cuda12x")
        time = np.arange(100., 11000., 2.)
        for field in (ScalarPlaneWave(.002, 1e-15*(1+.3j), K, .0007),
                      VectorPlaneWave(.002, [1e-15, 2e-15j, -3e-15], K, .0007)):
            cpu = BosonicFieldLinkResponse(orbit).compute(time, field)
            gpu = BosonicFieldLinkResponse(orbit, force_backend="cuda12x").compute(time, field)
            self.assertIsInstance(gpu.total, cp.ndarray)
            self.assertLess(np.linalg.norm(cp.asnumpy(gpu.total)-cpu.total)/np.linalg.norm(cpu.total), 1e-11)
            refined = BosonicFieldLinkResponse(orbit).compute(time, field, integration_step_s=30.)
            self.assertLess(np.linalg.norm(cpu.total-refined.total)/np.linalg.norm(refined.total), 1e-6)
            engine = FastLISAResponseTDI(orbits=orbit, order=15, tdi="2nd generation", tdi_chan="AE",
                                        force_backend="cuda12x", t_buffer=400., trim_garbage=True)
            result = engine.compute_links(time, gpu.total)
            reference = compute_tdi2_ae(time, cpu.total, orbits=orbit, t_buffer=400.)
            self.assertEqual(result.metadata["backend"], "fastlisaresponse_cuda12x")
            for name in ("A", "E"):
                values = cp.asnumpy(result.channels[name])
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertGreater(np.linalg.norm(values), 0.)
                self.assertLess(np.linalg.norm(values-reference.channels[name])/np.linalg.norm(reference.channels[name]), 1e-4)
