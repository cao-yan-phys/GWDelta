from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from gwdelta import C_SI, G_SI, SmoothVaidyaMassLoss, WeakFieldLinkResponse, build_link_geometry
from gwdelta.vaidya_response import profile_integrals, refine_geometry, synchronous_metric
from test_vaidya_response import make_orbits, crossing_model


def nonlinear_frequency_ratio(model, positions, velocity, geometry):
    velocity = np.broadcast_to(velocity, (3, 3))
    def force(t, x, v):
        h, hdot, gradient = synchronous_metric(model, t, x)
        spatial = hdot@v+np.einsum("ijk,j,k->i", gradient, v, v)
        spatial -= .5*np.einsum("jki,j,k->i", gradient, v, v)
        return -np.linalg.solve(np.eye(3)+h, spatial)+v*(v@hdot@v)/(2*C_SI**2)
    motions = []
    for sc in (0, 1):
        arrival = brentq(lambda t: t-np.linalg.norm(positions[sc]+velocity[sc]*t)/C_SI
                        -model.center_retarded_time_s, 0., 1300.)
        bounds = [0., arrival-24., arrival-8., arrival, arrival+8., arrival+24., 1300.]
        state = np.zeros(7)
        pieces = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            def rhs(t, y):
                position = positions[sc]+velocity[sc]*t+y[:3]
                actual_v = velocity[sc]+y[3:6]
                h = synchronous_metric(model, t, position)[0]
                sigma0 = np.sqrt(1-np.dot(velocity[sc], velocity[sc])/C_SI**2)
                sigma = np.sqrt(1-actual_v@(np.eye(3)+h)@actual_v/C_SI**2)
                clock_rate = -(2*np.dot(velocity[sc], y[3:6])+np.dot(y[3:6], y[3:6])+actual_v@h@actual_v)/(C_SI**2*(sigma+sigma0))
                return np.r_[y[3:6], force(t, position, actual_v), clock_rate]
            sol = solve_ivp(rhs, (a, b), state, method="DOP853", rtol=2e-11,
                            atol=1e-10, dense_output=True)
            if not sol.success:
                raise RuntimeError(sol.message)
            state = sol.y[:, -1]
            pieces.append(sol)
        def evaluate(t, sc=sc, bounds=bounds, pieces=pieces):
            index = np.clip(np.searchsorted(bounds, t, side="right")-1, 0, len(pieces)-1)
            y = pieces[index].sol(t)
            return positions[sc]+velocity[sc]*t+y[:3], velocity[sc]+y[3:6], y[6]
        motions.append(evaluate)
    sigma_r0 = np.sqrt(1-np.dot(velocity[0], velocity[0])/C_SI**2)
    tr = brentq(lambda t: sigma_r0*(t-1000.)+motions[0](t)[2], 999., 1001., xtol=1e-12)
    xr, vr, _clock = motions[0](tr)
    te0 = geometry.t_emission_s[0, 0]
    k = geometry.direction[0, 0]
    length = geometry.chord_length_m[0, 0]
    first = np.cross(k, [0., 0., 1.])
    first /= np.linalg.norm(first)
    second = np.cross(k, first)
    scale = 1e-6
    def shoot(parameters, output=False):
        te = te0+parameters[0]*scale*length/C_SI
        xe, ve, _clock = motions[1](te)
        direction = k+scale*(parameters[1]*first+parameters[2]*second)
        direction /= np.linalg.norm(direction)
        h = synchronous_metric(model, te, xe)[0]
        w0 = C_SI*direction/np.sqrt(direction@(np.eye(3)+h)@direction)
        initial = np.r_[np.zeros(3), (w0/C_SI-k)/scale, 0.]
        def rhs(s, y):
            w = C_SI*(k+scale*y[3:6])
            x = xe+C_SI*k*s+scale*length*y[:3]
            hdot = synchronous_metric(model, te+s, x)[1]
            return np.r_[C_SI/length*y[3:6], force(te+s, x, w)/(C_SI*scale),
                         -(w@hdot@w)/(2*C_SI**2*scale)]
        solution = solve_ivp(rhs, (0., tr-te), initial, method="DOP853", rtol=3e-11, atol=2e-11)
        if not solution.success:
            raise RuntimeError(solution.message)
        y = solution.y[:, -1]
        residual = (xe+C_SI*k*(tr-te)+scale*length*y[:3]-xr)/(scale*length)
        if output:
            return te, xe, ve, w0, C_SI*(k+scale*y[3:6]), scale*y[6], residual
        return residual
    solution = root(shoot, np.zeros(3), tol=1e-8)
    te, xe, ve, we, wr, log_energy, residual = shoot(solution.x, output=True)
    if np.linalg.norm(residual) > 2e-7:
        raise RuntimeError("nonlinear null-geodesic shooting failed to converge")
    he = synchronous_metric(model, te, xe)[0]
    hr = synchronous_metric(model, tr, xr)[0]
    ve0, vr0 = geometry.v_emission_m_s[0, 0], geometry.v_reception_m_s[0, 0]
    dve, dvr = ve-ve0, vr-vr0
    clock_e = -(2*np.dot(ve0, dve)+np.dot(dve, dve)+ve@he@ve)/C_SI**2
    clock_r = -(2*np.dot(vr0, dvr)+np.dot(dvr, dvr)+vr@hr@vr)/C_SI**2
    doppler_e = -(np.dot(dve, we)+np.dot(ve0, we-C_SI*k)+ve@he@we)/C_SI**2
    doppler_r = -(np.dot(dvr, wr)+np.dot(vr0, wr-C_SI*k)+vr@hr@wr)/C_SI**2
    log_ratio = log_energy+.5*np.log1p(clock_e/(1-np.dot(ve0, ve0)/C_SI**2))
    log_ratio -= .5*np.log1p(clock_r/(1-np.dot(vr0, vr0)/C_SI**2))
    log_ratio += np.log1p(doppler_r/(1-np.dot(k, vr0)/C_SI))
    log_ratio -= np.log1p(doppler_e/(1-np.dot(k, ve0)/C_SI))
    return np.expm1(log_ratio)


class VaidyaCovarianceTests(unittest.TestCase):
    def test_synchronous_curvature_matches_original_metric_reference(self):
        model = SmoothVaidyaMassLoss(6e25, 300., center_retarded_time_s=1000.)
        position = np.array([2.4e10, -1.1e10, .7e10])
        time = 1000.+np.linalg.norm(position)/C_SI+90.
        n, radius, f, fdot, _j, _q, _argument = profile_integrals(model, time, position)
        nn = np.outer(n, n)
        curvature = -G_SI*model.delta_mass_kg/C_SI**2*(
            fdot/(C_SI*radius**2)*(nn-np.eye(3))+f/radius**3*(np.eye(3)-3*nn)
        )
        reference = np.array([
            [1.927827537544521e-33, -1.391440855343603e-33, 8.854623624913837e-34],
            [-1.391440855343603e-33, -4.702996942029766e-34, -4.058369161418842e-34],
            [8.854623624913837e-34, -4.058369161418842e-34, -8.49783563842141e-34],
        ])
        np.testing.assert_allclose(curvature, reference, rtol=2e-13)

    def test_synchronous_metric_derivatives(self):
        model = SmoothVaidyaMassLoss(6e25, 300., center_retarded_time_s=1000.)
        position = np.array([2.4e10, -1.1e10, .7e10])
        time = 1000.+np.linalg.norm(position)/C_SI+90.
        _h, hdot, gradient = synchronous_metric(model, time, position)
        numerical_time = (synchronous_metric(model, time+.01, position)[0]
                          -synchronous_metric(model, time-.01, position)[0])/.02
        np.testing.assert_allclose(hdot, numerical_time, rtol=3e-9)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = 1e5
            numerical = (synchronous_metric(model, time, position+offset)[0]
                         -synchronous_metric(model, time, position-offset)[0])/2e5
            np.testing.assert_allclose(gradient[..., axis], numerical, rtol=3e-8)

    def test_moving_response_matches_nonlinear_null_and_timelike_geodesics(self):
        velocity = np.array([.03*C_SI, .01*C_SI, 0.])
        orbits, positions, _ = make_orbits(velocity)
        geometry = refine_geometry(orbits, build_link_geometry(orbits, np.array([1000.])))
        template = crossing_model(geometry, 1.)
        for mass in (1e33, 5e32):
            with self.subTest(mass=mass):
                model = SmoothVaidyaMassLoss(mass, 1., center_retarded_time_s=template.center_retarded_time_s)
                linear = WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model).as_numpy()["total"][0, 0]
                numerical = nonlinear_frequency_ratio(model, positions, velocity, geometry)
                self.assertLess(abs(numerical/linear-1), 2e-7)

    def test_different_endpoint_velocities_match_geodesic_shooting(self):
        velocity = C_SI*np.array([[.03, .01, 0.], [-.02, .015, .005], [.01, -.01, 0.]])
        orbits, positions, _ = make_orbits(velocity)
        geometry = refine_geometry(orbits, build_link_geometry(orbits, np.array([1000.])))
        template = crossing_model(geometry, 1.)
        model = SmoothVaidyaMassLoss(1e31, 1., center_retarded_time_s=template.center_retarded_time_s)
        linear = WeakFieldLinkResponse(orbits=orbits).compute(np.array([1000.]), model).as_numpy()["total"][0, 0]
        numerical = nonlinear_frequency_ratio(model, positions, velocity, geometry)
        self.assertLess(abs(numerical/linear-1), 5e-7)
