from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from gwdelta import C_SI, G_SI, SampledOrbits, SmoothVaidyaMassLoss, WeakFieldLinkResponse, build_link_geometry


def make_orbits(velocity=None, distance=1.495978707e11):
    angles = np.deg2rad([0., 120., 240.])
    positions = np.array([distance, -2e10, 1e10]) + 2.5e9 / np.sqrt(3.) * np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros(3))
    )
    time = np.array([0., 2000.])
    v = np.zeros((3, 3)) if velocity is None else np.broadcast_to(velocity, (3, 3))
    x = positions[None] + time[:, None, None] * v[None]
    return SampledOrbits(time, x, armlength=2.5e9, force_backend="cpu"), positions, v


def crossing_model(geometry, tau):
    xe, xr = geometry.x_emission_m[0, 0], geometry.x_reception_m[0, 0]
    te, lt = geometry.t_emission_s[0, 0], geometry.light_time_s[0, 0]
    midpoint = xe + .43 * (xr-xe)
    u0 = te + .43*lt - np.linalg.norm(midpoint)/C_SI
    return SmoothVaidyaMassLoss(1e25, tau, center_retarded_time_s=u0)


def fixed_velocity(model, time, position, initial_time):
    radius = np.linalg.norm(position, axis=-1)
    q = (np.asarray(time)-radius/C_SI-model.center_retarded_time_s)/model.transition_time_s
    q0 = (initial_time-radius/C_SI-model.center_retarded_time_s)/model.transition_time_s
    f = .5*(1+np.tanh(q))
    f0 = .5*(1+np.tanh(q0))
    integral = .5*model.transition_time_s*(np.logaddexp(0., 2*q)-np.logaddexp(0., 2*q0))
    scale = G_SI*model.delta_mass_kg*(integral/radius**2-(f-f0)/(C_SI*radius))
    return scale[..., None]*position/radius[..., None]


def direct_reference(model, geometry, link=0, sample=0):
    xe = geometry.x_emission_m[link, sample]
    xr = geometry.x_reception_m[link, sample]
    te = geometry.t_emission_s[link, sample]
    tr = geometry.t_reception_s[sample]
    lt = geometry.light_time_s[link, sample]
    k = geometry.direction[link, sample]
    scale = G_SI*model.delta_mass_kg/(C_SI**2*np.linalg.norm(xe))
    def argument(z):
        return (te+lt*z-np.linalg.norm(xe+(xr-xe)*z)/C_SI-model.center_retarded_time_s)/model.transition_time_s
    breaks = [0., 1.]
    for target in (-30., -10., -3., -1., 0., 1., 3., 10., 30.):
        if argument(0.) < target < argument(1.):
            breaks.append(brentq(lambda z: argument(z)-target, 0., 1.))
    breaks.sort()
    def integrand(z):
        position = xe+(xr-xe)*z
        radius = np.linalg.norm(position)
        mu = np.dot(k, position/radius)
        return lt*float(model.time_derivative(te+lt*z, position).psi)/scale*(1-mu)**2
    propagation = sum(quad(integrand, a, b, epsabs=2e-13, epsrel=2e-12)[0]
                      for a, b in zip(breaks[:-1], breaks[1:]))
    return float(model.metric(te, xe).psi-model.metric(tr, xr).psi)+scale*propagation


class VaidyaResponseTests(unittest.TestCase):
    def test_complete_response_resolves_thin_shell_and_cancellation(self):
        orbits, positions, _ = make_orbits()
        time = np.array([1000.])
        geometry = build_link_geometry(orbits, time)
        for tau in (1., .01, .001):
            with self.subTest(tau=tau):
                model = crossing_model(geometry, tau)
                result = WeakFieldLinkResponse(orbits=orbits).compute(time, model)
                direct = direct_reference(model, geometry)
                k = geometry.direction[0, 0]
                endpoint = -np.dot(k, fixed_velocity(model, time[0], positions[0], 0.)
                    -fixed_velocity(model, geometry.t_emission_s[0, 0], positions[1], 0.))/C_SI
                expected = direct+endpoint
                self.assertLess(abs(result.as_numpy()["total"][0, 0]/expected-1), 2e-7)
                diagnostic = WeakFieldLinkResponse(orbits=orbits).compute(time, model, include_endpoint_motion=False)
                self.assertLess(abs(diagnostic.as_numpy()["direct"][0, 0]/direct-1), 2e-10)
                self.assertEqual(result.metadata["worldline_velocity_source"], "automatic_vaidya")
                self.assertEqual(result.metadata["response_gauge"], "synchronous")

    def test_three_spacecraft_dense_velocity_with_coarse_support(self):
        orbits, positions, _ = make_orbits()
        model = crossing_model(build_link_geometry(orbits, np.array([1000.])), .001)
        support = np.array([0., 2000.])
        background = np.broadcast_to(positions, (2, 3, 3))
        motion = model.integrate_test_mass_motion(support, background)
        arrival = model.center_retarded_time_s+np.linalg.norm(positions, axis=-1)/C_SI
        for sc in range(3):
            query = arrival[sc]+np.array([-.001, 0., .001, 1.])
            np.testing.assert_array_equal(motion.velocity_evaluator(query)[:, sc], np.zeros((len(query), 3)))
            np.testing.assert_array_equal(motion.position_evaluator(query)[:, sc], np.zeros((len(query), 3)))
        self.assertEqual(motion.delta_velocity_m_s.shape, (2, 3, 3))

    def test_moving_background_matches_synchronous_geodesic_integral(self):
        from gwdelta.vaidya_response import motion_force
        orbits, positions, velocity = make_orbits(np.array([10000., -5000., 1000.]))
        geometry = build_link_geometry(orbits, np.array([1000.]))
        model = crossing_model(geometry, .1)
        support = np.array([0., 2000.])
        motion = model.integrate_test_mass_motion(support, positions[None]+support[:, None, None]*velocity[None])
        query = np.array([995., 1000., 1005.])
        for sc in range(3):
            def arg(t):
                return t-np.linalg.norm(positions[sc]+velocity[sc]*t)/C_SI-model.center_retarded_time_s
            arrival = brentq(arg, 0., 2000.)
            breaks = sorted([0., arrival-3., arrival, arrival+3., 1010.])
            state = np.zeros(3)
            solutions = []
            scale = G_SI*model.delta_mass_kg/(C_SI*np.linalg.norm(positions[sc]))
            for a, b in zip(breaks[:-1], breaks[1:]):
                sol = solve_ivp(lambda t, y: motion_force(model, t, positions[sc]+velocity[sc]*t, velocity[sc])/scale,
                    (a,b), state, method="DOP853", rtol=2e-12, atol=2e-14, dense_output=True)
                self.assertTrue(sol.success)
                state = sol.y[:, -1]
                solutions.append(sol)
            reference = np.array([solutions[np.clip(np.searchsorted(breaks,t,side="right")-1,0,len(solutions)-1)].sol(t)*scale for t in query])
            np.testing.assert_allclose(motion.velocity_evaluator(query)[:, sc], reference, rtol=2e-9, atol=1e-18)
        result = WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model)
        np.testing.assert_allclose(result.as_numpy()["total"], result.as_numpy()["direct"]+result.as_numpy()["worldline"], rtol=2e-8, atol=1e-25)

    def test_missing_initial_conditions_are_not_silently_assumed(self):
        orbits, positions, _ = make_orbits()
        model = SmoothVaidyaMassLoss(1e25, 1., center_retarded_time_s=-1e6)
        with self.assertRaisesRegex(ValueError, "pre-loss"):
            WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model)
        with self.assertRaisesRegex(ValueError, "initial conditions"):
            model.integrate_test_mass_motion(np.array([0., 2000.]), np.broadcast_to(positions, (2, 3, 3)))
        diagnostic = WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model, include_endpoint_motion=False)
        self.assertFalse(diagnostic.metadata["worldline_velocity_included"])

    def test_kiloparsec_source_matches_high_precision_reference(self):
        orbits, _, _ = make_orbits()
        time = np.array([1000.])
        geometry = build_link_geometry(orbits, time)
        origin = np.array([-3.085677581491367e19, 0., 0.])
        xe = geometry.x_emission_m[0, 0]
        xr = geometry.x_reception_m[0, 0]
        crossing = xe+.43*(xr-xe)
        u0 = time[0]-.57*geometry.light_time_s[0, 0]-np.linalg.norm(crossing-origin)/C_SI
        model = SmoothVaidyaMassLoss(1e25, .01, center_retarded_time_s=u0, origin_m=tuple(origin))
        result = WeakFieldLinkResponse(orbits=orbits).compute(time, model)
        reference_60_digits = 2.7786094645706773077e-33
        self.assertLess(abs(result.as_numpy()["total"][0, 0]/reference_60_digits-1), 2e-9)

    def test_explicit_initial_velocity_and_model_identity(self):
        orbits, positions, _ = make_orbits()
        support = np.array([0., 2000.])
        background = np.broadcast_to(positions, (2, 3, 3))
        model = SmoothVaidyaMassLoss(1e25, 1., center_retarded_time_s=-1e6)
        initial = np.array([[1e-5, 2e-5, 3e-5], [4e-5, 5e-5, 6e-5], [7e-5, 8e-5, 9e-5]])
        motion = model.integrate_test_mass_motion(support, background, initial_delta_velocity_m_s=initial)
        np.testing.assert_allclose(motion.velocity_evaluator(support[:1])[0], initial, rtol=2e-12)
        response = WeakFieldLinkResponse(orbits=orbits)
        result = response.compute(np.array([1000.]), model, delta_velocity_m_s=motion)
        self.assertTrue(np.all(np.isfinite(result.as_numpy()["total"])))
        with self.assertRaisesRegex(ValueError, "synchronous gauge"):
            response.compute(np.array([1000.]), SmoothVaidyaMassLoss(2e25, 1.), delta_velocity_m_s=motion)

    def test_moving_far_source_has_no_planar_gauge_signal(self):
        for velocity in (np.array([30000., 0., 0.]), np.array([0., 30000., 0.])):
            orbits, _, _ = make_orbits(velocity)
            geometry = build_link_geometry(orbits, np.array([1000.]))
            amplitudes = []
            for distance in (1e16, 1e18):
                origin = np.array([-distance, 0., 0.])
                xe, xr = geometry.x_emission_m[0, 0], geometry.x_reception_m[0, 0]
                crossing = xe+.43*(xr-xe)
                u0 = 1000.-.57*geometry.light_time_s[0, 0]-np.linalg.norm(crossing-origin)/C_SI
                model = SmoothVaidyaMassLoss(1e25, .1, center_retarded_time_s=u0, origin_m=tuple(origin))
                result = WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model)
                amplitudes.append(result.as_numpy()["total"][0, 0])
            self.assertLess(abs(amplitudes[0]/amplitudes[1]/1e4-1), .002)

    def test_background_feedback_and_position_solution(self):
        _orbits, positions, _ = make_orbits()
        model = SmoothVaidyaMassLoss(1e25, 1., center_retarded_time_s=1e6)
        support = np.array([0., 2000.])
        initial = np.array([1., 0., 0.])
        omega = .001
        motion = model.integrate_test_mass_motion(
            support, np.broadcast_to(positions, (2, 3, 3)),
            initial_delta_position_m=initial,
            background_acceleration_jacobian=lambda t, x, v: -omega**2*np.eye(3),
        )
        query = np.array([100., 500., 1500.])
        np.testing.assert_allclose(motion.position_evaluator(query)[:, 0, 0], np.cos(omega*query), rtol=2e-10)
        np.testing.assert_allclose(motion.velocity_evaluator(query)[:, 0, 0], -omega*np.sin(omega*query), rtol=2e-10)

    def test_photon_chord_through_source_is_rejected(self):
        orbits, _, _ = make_orbits()
        geometry = build_link_geometry(orbits, np.array([1000.]))
        origin = .5*(geometry.x_emission_m[0, 0]+geometry.x_reception_m[0, 0])
        model = SmoothVaidyaMassLoss(1e25, 20., center_retarded_time_s=700., origin_m=tuple(origin))
        with self.assertRaisesRegex(ValueError, "source origin"):
            WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model)

    def test_post_shell_response_with_long_time_history(self):
        _orbits, positions, _ = make_orbits()
        support = np.array([0., 2e6])
        orbits = SampledOrbits(support, np.broadcast_to(positions, (2, 3, 3)), armlength=2.5e9, force_backend="cpu")
        time = np.array([1e6])
        geometry = build_link_geometry(orbits, time)
        model = SmoothVaidyaMassLoss(1e25, 1., center_retarded_time_s=500.)
        xe, xr = geometry.x_emission_m[0, 0], geometry.x_reception_m[0, 0]
        expected = float(model.metric(geometry.t_emission_s[0, 0], xe).psi-model.metric(time[0], xr).psi)
        expected -= np.dot(geometry.direction[0, 0],
            fixed_velocity(model, time[0], xr, 0.)-fixed_velocity(model, geometry.t_emission_s[0, 0], xe, 0.))/C_SI
        result = WeakFieldLinkResponse(orbits=orbits).compute(time, model).as_numpy()["total"][0, 0]
        self.assertLess(abs(result/expected-1), 2e-11)


if __name__ == "__main__":
    unittest.main()
