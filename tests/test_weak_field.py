from __future__ import annotations

import unittest

import numpy as np

from gwdelta import (
    C_SI,
    G_SI,
    MetricValues,
    RetardedQuadrupoleMode,
    SampledOrbits,
    SmoothVaidyaMassLoss,
    UniformMovingPointMass,
    WeakFieldLinkResponse,
    build_link_geometry,
    link_fd_polarization_response,
    link_fd_response,
    polarization_tensors,
    sky_basis,
)


def static_orbits(
    *, arm_m: float = 2.5e9, duration_s: float = 40000.0
) -> tuple[SampledOrbits, np.ndarray]:
    center = np.asarray([1.495978707e11, -2.0e10, 1.0e10])
    radius = arm_m / np.sqrt(3.0)
    angles = np.deg2rad([0.0, 120.0, 240.0])
    positions = center + np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(3)]
    )
    times = np.asarray([0.0, duration_s])
    position_series = np.repeat(positions[np.newaxis, :, :], len(times), axis=0)
    return (
        SampledOrbits(times, position_series, armlength=arm_m, force_backend="cpu"),
        positions,
    )


class PlaneTensorField:
    phasor_sign = 1

    def __init__(self, frequency_hz: float, propagation, tensor) -> None:
        self.angular_frequency = 2.0 * np.pi * float(frequency_hz)
        self.propagation = np.asarray(propagation, dtype=float)
        self.tensor = np.asarray(tensor, dtype=float)

    def complex_amplitude(self, x_m) -> MetricValues:
        x = np.asarray(x_m)
        phase = np.exp(
            -1j
            * self.angular_frequency
            * np.einsum("...i,i->...", x, self.propagation)
            / C_SI
        )
        zeros = np.zeros(phase.shape)
        return MetricValues(
            psi=zeros.astype(complex),
            xi=np.zeros(phase.shape + (3,), dtype=complex),
            h=phase[..., np.newaxis, np.newaxis] * self.tensor,
        )

    def metric(self, t_s, x_m) -> MetricValues:
        amplitude = self.complex_amplitude(x_m)
        phase = np.exp(1j * self.angular_frequency * np.asarray(t_s))
        return MetricValues(
            psi=np.real(amplitude.psi * phase),
            xi=np.real(amplitude.xi * phase[..., np.newaxis]),
            h=np.real(amplitude.h * phase[..., np.newaxis, np.newaxis]),
        )

    def time_derivative(self, t_s, x_m) -> MetricValues:
        amplitude = self.complex_amplitude(x_m)
        phase = (
            1j
            * self.angular_frequency
            * np.exp(1j * self.angular_frequency * np.asarray(t_s))
        )
        return MetricValues(
            psi=np.real(amplitude.psi * phase),
            xi=np.real(amplitude.xi * phase[..., np.newaxis]),
            h=np.real(amplitude.h * phase[..., np.newaxis, np.newaxis]),
        )


class UniformTimePotential:
    def __init__(self, slope_per_s: float) -> None:
        self.slope = float(slope_per_s)

    def metric(self, t_s, x_m) -> MetricValues:
        t = np.asarray(t_s)
        return MetricValues(
            psi=self.slope * t,
            xi=np.zeros(t.shape + (3,)),
            h=np.zeros(t.shape + (3, 3)),
        )

    def time_derivative(self, t_s, x_m) -> MetricValues:
        t = np.asarray(t_s)
        return MetricValues(
            psi=np.full(t.shape, self.slope),
            xi=np.zeros(t.shape + (3,)),
            h=np.zeros(t.shape + (3, 3)),
        )


class NumericalMovingPointMass:
    """Hide the analytic hook to exercise the general path integrator."""

    def __init__(self, source: UniformMovingPointMass) -> None:
        self.source = source

    def metric(self, t_s, x_m) -> MetricValues:
        return self.source.metric(t_s, x_m)

    def time_derivative(self, t_s, x_m) -> MetricValues:
        return self.source.time_derivative(t_s, x_m)


class WeakFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orbits, cls.positions = static_orbits()

    def test_uniform_time_potential_cancels(self) -> None:
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=8, force_backend="cpu"
        )
        result = engine.compute(
            np.linspace(0.0, 2000.0, 41), UniformTimePotential(2.0e-12)
        )
        np.testing.assert_allclose(result.as_numpy()["direct"], 0.0, atol=2.0e-25)

    def test_endpoint_velocity_term_uses_receiver_emitter_order(self) -> None:
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=8, force_backend="cpu"
        )
        times = np.linspace(0.0, 2000.0, 41)
        spacecraft_velocity = np.asarray(
            [[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0], [0.0, -1.0, 4.0]]
        )
        delta_velocity = np.repeat(
            spacecraft_velocity[np.newaxis, :, :], len(times), axis=0
        )
        result = engine.compute(
            times,
            UniformTimePotential(0.0),
            delta_velocity_m_s=delta_velocity,
        )
        receivers = np.asarray([0, 1, 2, 0, 2, 1])
        emitters = np.asarray([1, 2, 0, 2, 1, 0])
        expected = (
            -np.einsum(
                "lti,li->lt",
                result.geometry.direction,
                spacecraft_velocity[receivers] - spacecraft_velocity[emitters],
            )
            / C_SI
        )
        np.testing.assert_allclose(
            result.as_numpy()["worldline"], expected, rtol=2.0e-14, atol=0.0
        )

    def test_plane_wave_matches_existing_fd_link_response(self) -> None:
        frequency = 0.003
        lam = 0.3
        beta = 0.4
        propagation, u_basis, v_basis = sky_basis(lam, beta)
        plus_tensor = np.outer(u_basis, u_basis) - np.outer(v_basis, v_basis)
        field = PlaneTensorField(frequency, propagation, plus_tensor)
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=48, force_backend="cpu"
        )

        actual = engine.compute_monochromatic(1000.0, field).as_numpy()
        expected = link_fd_response(
            np.asarray([frequency]),
            lam=lam,
            beta=beta,
            positions_m=self.positions,
        )[0][:, 0]
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-14)

    def test_all_spatial_polarizations_match_fd_link_response(self) -> None:
        frequency = 0.003
        lam = -0.5
        beta = 0.2
        propagation, _u_basis, _v_basis = sky_basis(lam, beta)
        expected = link_fd_polarization_response(
            np.asarray([frequency]),
            lam=lam,
            beta=beta,
            positions_m=self.positions,
        )
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=48, force_backend="cpu"
        )
        for name, tensor in polarization_tensors(lam, beta).items():
            field = PlaneTensorField(frequency, propagation, tensor)
            actual = engine.compute_monochromatic(1000.0, field).as_numpy()
            np.testing.assert_allclose(
                actual, expected[name][:, 0], rtol=2.0e-13, atol=2.0e-14
            )

    def test_quadrupole_time_domain_matches_complex_amplitude(self) -> None:
        frequency = 3.0e-4
        omega = 2.0 * np.pi * frequency
        quadrupole = 1.0e45 * np.asarray(
            [[1.0, 0.2, 0.0], [0.2, -0.4, 0.1], [0.0, 0.1, -0.6]],
            dtype=complex,
        )
        field = RetardedQuadrupoleMode(omega, quadrupole)
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=32, force_backend="cpu"
        )
        times = np.linspace(0.0, 20000.0, 401)

        actual = engine.compute(times, field).as_numpy()["direct"]
        amplitude = engine.compute_monochromatic(1000.0, field).as_numpy()
        expected = np.real(
            amplitude[:, np.newaxis] * np.exp(-1j * omega * times[np.newaxis, :])
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=1.0e-28)

    def test_quadrupole_near_and_wave_zone_limits(self) -> None:
        omega = 2.0 * np.pi * 1.0e-4
        quadrupole = 1.0e35 * np.diag([1.0, -1.0, 0.0]).astype(complex)
        field = RetardedQuadrupoleMode(omega, quadrupole)

        x_near = np.asarray([[1.0e6, 0.0, 0.0]])
        actual_psi = field.complex_amplitude(x_near).psi[0]
        radial = x_near[0] / np.linalg.norm(x_near[0])
        retarded_quadrupole = quadrupole * np.exp(
            1j * omega * np.linalg.norm(x_near[0]) / C_SI
        )
        expected_psi = (
            -3.0
            * G_SI
            / (2.0 * C_SI**2 * np.linalg.norm(x_near[0]) ** 3)
            * np.einsum("i,j,ij", radial, radial, retarded_quadrupole)
        )
        self.assertLess(abs(actual_psi - expected_psi) / abs(expected_psi), 3.0e-6)

        x_wave = np.asarray([[0.0, 0.0, 1.0e15]])
        metric = field.complex_amplitude(x_wave).h[0]
        projector = np.diag([1.0, 1.0, 0.0])
        metric_tt = projector @ metric @ projector - 0.5 * projector * np.trace(
            projector @ metric
        )
        radius = np.linalg.norm(x_wave[0])
        i_ddot = -(omega**2) * quadrupole * np.exp(1j * omega * radius / C_SI)
        expected = 2.0 * G_SI / (C_SI**4 * radius) * i_ddot
        expected_tt = projector @ expected @ projector - 0.5 * projector * np.trace(
            projector @ expected
        )
        np.testing.assert_allclose(metric_tt, expected_tt, rtol=2.0e-12, atol=1.0e-40)

    def test_quadrupole_expansion_matches_retarded_spatial_derivatives(self) -> None:
        omega = 2.0 * np.pi * 3.0e-4
        quadrupole = 1.0e42 * np.asarray(
            [[1.0, 0.2, 0.1], [0.2, -0.4, 0.05], [0.1, 0.05, -0.6]],
            dtype=complex,
        )
        field = RetardedQuadrupoleMode(omega, quadrupole)
        position = np.asarray([1.2e11, -4.0e10, 3.0e10])
        step = 1.0e7
        basis = np.eye(3)

        def retarded_quadrupole(x):
            radius = np.linalg.norm(x)
            return quadrupole * np.exp(1j * omega * radius / C_SI)

        second_derivative = 0.0j
        for a in range(3):
            for b in range(3):
                def component(x):
                    return retarded_quadrupole(x)[a, b] / np.linalg.norm(x)

                if a == b:
                    second_derivative += (
                        component(position + step * basis[a])
                        - 2.0 * component(position)
                        + component(position - step * basis[a])
                    ) / step**2
                else:
                    second_derivative += (
                        component(position + step * basis[a] + step * basis[b])
                        - component(position + step * basis[a] - step * basis[b])
                        - component(position - step * basis[a] + step * basis[b])
                        + component(position - step * basis[a] - step * basis[b])
                    ) / (4.0 * step**2)

        first_derivative = np.zeros(3, dtype=complex)
        for a in range(3):
            for b in range(3):
                def dotted_component(x):
                    return (
                        -1j
                        * omega
                        * retarded_quadrupole(x)[a, b]
                        / np.linalg.norm(x)
                    )

                first_derivative[a] += (
                    dotted_component(position + step * basis[b])
                    - dotted_component(position - step * basis[b])
                ) / (2.0 * step)

        metric = field.complex_amplitude(position)
        expected_h00 = G_SI / C_SI**2 * second_derivative
        expected_h0 = 2.0 * G_SI / C_SI**3 * first_derivative
        np.testing.assert_allclose(-2.0 * metric.psi, expected_h00, rtol=5.0e-8)
        np.testing.assert_allclose(metric.xi, expected_h0, rtol=5.0e-8)

    def test_quadrupole_test_mass_acceleration_matches_metric_gradient(self) -> None:
        omega = 2.0 * np.pi * 3.0e-4
        quadrupole = 1.0e42 * np.asarray(
            [[1.0, 0.2, 0.1], [0.2, -0.4, 0.05], [0.1, 0.05, -0.6]],
            dtype=complex,
        )
        field = RetardedQuadrupoleMode(omega, quadrupole)
        position = np.asarray([1.2e11, -4.0e10, 3.0e10])
        step = 1.0e6
        gradient = np.empty(3, dtype=complex)
        for axis in range(3):
            displacement = np.zeros(3)
            displacement[axis] = step
            gradient[axis] = (
                field.complex_amplitude(position + displacement).psi
                - field.complex_amplitude(position - displacement).psi
            ) / (2.0 * step)
        expected = (
            -C_SI**2 * gradient
            + 1j * C_SI * omega * field.complex_amplitude(position).xi
        )
        actual = field.acceleration_complex_amplitude(position)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-8, atol=1.0e-24)

        times = np.asarray([0.0, 100.0, 200.0])
        positions = np.repeat(position[np.newaxis, np.newaxis, :], 3, axis=0)
        motion = field.steady_state_test_mass_motion(times, positions).as_numpy()
        phase = np.exp(-1j * omega * times)[:, np.newaxis, np.newaxis]
        expected_acceleration = np.real(actual[np.newaxis, np.newaxis, :] * phase)
        expected_velocity = np.real(
            1j * actual[np.newaxis, np.newaxis, :] * phase / omega
        )
        np.testing.assert_allclose(
            motion["acceleration_m_s2"], expected_acceleration, rtol=2.0e-14
        )
        np.testing.assert_allclose(
            motion["delta_velocity_m_s"], expected_velocity, rtol=2.0e-14
        )

    def test_quadrupole_and_vaidya_parameters_are_strict(self) -> None:
        quadrupole = np.diag([1.0, -1.0, 0.0]).astype(complex)
        with self.assertRaises(ValueError):
            RetardedQuadrupoleMode(
                1.0, quadrupole * np.nan, origin_m=(0.0, 0.0, 0.0)
            )
        with self.assertRaises(ValueError):
            RetardedQuadrupoleMode(1.0, quadrupole, origin_m=(0.0, np.inf, 0.0))
        with self.assertRaises(ValueError):
            SmoothVaidyaMassLoss(np.nan, 1.0)
        with self.assertRaises(ValueError):
            SmoothVaidyaMassLoss(1.0, np.inf)
        with self.assertRaises(ValueError):
            SmoothVaidyaMassLoss(1.0, 1.0, center_retarded_time_s=np.nan)
        with self.assertRaises(ValueError):
            SmoothVaidyaMassLoss(1.0, 1.0, origin_m=(0.0, 0.0))

    def test_vaidya_metric_has_expected_kernel(self) -> None:
        field = SmoothVaidyaMassLoss(
            delta_mass_kg=1.0e25,
            transition_time_s=20.0,
            center_retarded_time_s=100.0,
        )
        t = np.asarray([610.0, 620.0])
        x = np.asarray([[1.5e11, 2.0e9, 0.0], [1.5e11, -3.0e9, 1.0e9]])
        direction = np.asarray([[0.2, 0.9, 0.3], [-0.4, 0.1, 0.8]])
        direction /= np.linalg.norm(direction, axis=-1)[:, np.newaxis]
        values = field.metric(t, x)
        kernel = (
            values.psi
            - np.einsum("...i,...i->...", direction, values.xi)
            - 0.5 * np.einsum("...i,...ij,...j->...", direction, values.h, direction)
        )
        radial = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
        mu = np.einsum("...i,...i->...", direction, radial)
        expected = values.psi * (1.0 - mu) ** 2
        np.testing.assert_allclose(kernel, expected, rtol=2.0e-14, atol=0.0)

    def test_vaidya_is_mass_loss_relative_to_the_pre_loss_field(self) -> None:
        mass_lost = 2.0e25
        field = SmoothVaidyaMassLoss(
            delta_mass_kg=mass_lost,
            transition_time_s=2.0,
            center_retarded_time_s=100.0,
        )
        position = np.asarray([[3.0e8, 0.0, 0.0]])
        early = field.metric(np.asarray([61.0]), position)
        late = field.metric(np.asarray([141.0]), position)
        expected = G_SI * mass_lost / (C_SI**2 * np.linalg.norm(position[0]))
        self.assertLess(abs(early.psi[0]), expected * 1.0e-12)
        np.testing.assert_allclose(late.psi, expected, rtol=1.0e-12)
        np.testing.assert_allclose(late.xi[0], [2.0 * expected, 0.0, 0.0])
        np.testing.assert_allclose(
            late.h[0], np.diag([-2.0 * expected, 0.0, 0.0])
        )

    def test_vaidya_late_time_link_is_static_mass_difference(self) -> None:
        mass_lost = 3.0e25
        field = SmoothVaidyaMassLoss(
            delta_mass_kg=mass_lost,
            transition_time_s=1.0,
            center_retarded_time_s=-1.0e6,
        )
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=12, force_backend="cpu"
        )
        result = engine.compute(np.asarray([1000.0]), field)
        receiver_radius = np.linalg.norm(result.geometry.x_reception_m[:, 0], axis=-1)
        emitter_radius = np.linalg.norm(result.geometry.x_emission_m[:, 0], axis=-1)
        expected = G_SI * mass_lost / C_SI**2 * (
            1.0 / emitter_radius - 1.0 / receiver_radius
        )
        np.testing.assert_allclose(
            result.as_numpy()["direct"][:, 0], expected, rtol=3.0e-14
        )

    def test_vaidya_derivative_and_test_mass_acceleration(self) -> None:
        field = SmoothVaidyaMassLoss(
            delta_mass_kg=6.0e25,
            transition_time_s=300.0,
            center_retarded_time_s=1000.0,
        )
        position = np.asarray([2.4e10, -1.1e10, 0.7e10])
        radius = np.linalg.norm(position)
        time = 1000.0 + radius / C_SI + 90.0
        time_step = 1.0e-2
        position_step = 1.0e5

        plus = field.metric(time + time_step, position)
        minus = field.metric(time - time_step, position)
        derivative = field.time_derivative(time, position)
        np.testing.assert_allclose(
            derivative.psi,
            (plus.psi - minus.psi) / (2.0 * time_step),
            rtol=2.0e-9,
        )
        np.testing.assert_allclose(
            derivative.xi,
            (plus.xi - minus.xi) / (2.0 * time_step),
            rtol=2.0e-9,
        )
        np.testing.assert_allclose(
            derivative.h,
            (plus.h - minus.h) / (2.0 * time_step),
            rtol=2.0e-9,
        )

        gradient = np.empty(3)
        for index in range(3):
            offset = np.zeros(3)
            offset[index] = position_step
            gradient[index] = (
                field.metric(time, position + offset).psi
                - field.metric(time, position - offset).psi
            ) / (2.0 * position_step)
        acceleration_from_metric = -C_SI**2 * gradient - C_SI * derivative.xi
        np.testing.assert_allclose(
            field.acceleration(time, position),
            acceleration_from_metric,
            rtol=2.0e-9,
            atol=1.0e-24,
        )

    def test_vaidya_test_mass_motion_matches_analytic_shell_integral(self) -> None:
        mass_lost = 2.0e25
        transition_time = 50.0
        radius = 1.0e10
        field = SmoothVaidyaMassLoss(
            delta_mass_kg=mass_lost,
            transition_time_s=transition_time,
        )
        retarded_time = np.linspace(
            -20.0 * transition_time, 20.0 * transition_time, 20001
        )
        times = radius / C_SI + retarded_time
        positions = np.zeros((len(times), 3))
        positions[:, 0] = radius
        motion = field.integrate_test_mass_motion(times, positions).as_numpy()

        def antiderivative_profile(u):
            return 0.5 * (u + transition_time * np.log(np.cosh(u / transition_time)))

        def profile(u):
            return 0.5 * (1.0 + np.tanh(u / transition_time))

        expected_velocity = G_SI * mass_lost * (
            (antiderivative_profile(retarded_time[-1])
             - antiderivative_profile(retarded_time[0]))
            / radius**2
            - (profile(retarded_time[-1]) - profile(retarded_time[0]))
            / (C_SI * radius)
        )
        np.testing.assert_allclose(
            motion["delta_velocity_m_s"][-1],
            [expected_velocity, 0.0, 0.0],
            rtol=2.0e-8,
            atol=1.0e-20,
        )

    def test_moving_point_mass_parameters_and_metric_derivative(self) -> None:
        velocity = C_SI * np.asarray([0.12, -0.04, 0.03])
        source = UniformMovingPointMass(
            rest_mass_kg=2.0e25,
            position_at_reference_m=(2.0e9, -3.0e9, 1.0e9),
            velocity_m_s=tuple(velocity),
            reference_time_s=50.0,
        )
        np.testing.assert_allclose(
            source.position(50.0), source.position_at_reference_m
        )
        np.testing.assert_allclose(source.beta_vector, velocity / C_SI)
        self.assertAlmostEqual(source.beta, np.linalg.norm(velocity) / C_SI)

        times = np.asarray([20.0, 80.0])
        positions = np.asarray([[5.0e9, 2.0e9, -4.0e9], [6.0e9, -1.0e9, 3.0e9]])
        step = 1.0e-3
        plus = source.metric(times + step, positions)
        minus = source.metric(times - step, positions)
        derivative = source.time_derivative(times, positions)
        np.testing.assert_allclose(
            derivative.psi, (plus.psi - minus.psi) / (2.0 * step), rtol=2.0e-6
        )
        np.testing.assert_allclose(
            derivative.xi, (plus.xi - minus.xi) / (2.0 * step), rtol=2.0e-6
        )
        np.testing.assert_allclose(
            derivative.h, (plus.h - minus.h) / (2.0 * step), rtol=2.0e-6
        )

        with self.assertRaises(ValueError):
            UniformMovingPointMass(
                rest_mass_kg=1.0,
                position_at_reference_m=(0.0, 0.0, 0.0),
                velocity_m_s=(C_SI, 0.0, 0.0),
            )

    def test_moving_point_mass_static_limit(self) -> None:
        source_position = np.mean(self.positions, axis=0) + np.asarray(
            [7.0e9, -4.0e9, 3.0e9]
        )
        source = UniformMovingPointMass(
            rest_mass_kg=3.0e25,
            position_at_reference_m=tuple(source_position),
            velocity_m_s=(0.0, 0.0, 0.0),
        )
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=8, force_backend="cpu"
        )
        result = engine.compute(np.asarray([1000.0]), source)
        receiver_radius = np.linalg.norm(
            result.geometry.x_reception_m[:, 0] - source_position, axis=-1
        )
        emitter_radius = np.linalg.norm(
            result.geometry.x_emission_m[:, 0] - source_position, axis=-1
        )
        expected = (
            G_SI
            * source.rest_mass_kg
            / C_SI**2
            * (1.0 / receiver_radius - 1.0 / emitter_radius)
        )
        np.testing.assert_allclose(
            result.as_numpy()["direct"][:, 0], expected, rtol=2.0e-14
        )
        self.assertEqual(result.metadata["direct_evaluation"], "analytic")

    def test_moving_point_mass_analytic_links_match_general_integrator(self) -> None:
        source = UniformMovingPointMass(
            rest_mass_kg=8.0e25,
            position_at_reference_m=tuple(
                np.mean(self.positions, axis=0) + np.asarray([5.0e9, 4.0e9, 3.0e9])
            ),
            velocity_m_s=tuple(C_SI * np.asarray([0.08, -0.03, 0.02])),
            reference_time_s=1000.0,
        )
        times = np.linspace(920.0, 1080.0, 33)
        engine = WeakFieldLinkResponse(
            orbits=self.orbits, quadrature_order=160, force_backend="cpu"
        )
        analytic = engine.compute(times, source).as_numpy()["direct"]
        numerical = engine.compute(times, NumericalMovingPointMass(source)).as_numpy()[
            "direct"
        ]
        np.testing.assert_allclose(analytic, numerical, rtol=3.0e-11, atol=1.0e-24)

    def test_moving_point_mass_worldline_kick(self) -> None:
        beta = 0.2
        impact_parameter = 1.0e9
        source = UniformMovingPointMass(
            rest_mass_kg=1.0e25,
            position_at_reference_m=(0.0, 0.0, 0.0),
            velocity_m_s=(beta * C_SI, 0.0, 0.0),
            reference_time_s=0.0,
        )
        characteristic_time = impact_parameter / (source.lorentz_factor * beta * C_SI)
        times = np.linspace(
            -500.0 * characteristic_time, 500.0 * characteristic_time, 100001
        )
        positions = np.zeros((len(times), 3))
        positions[:, 1] = impact_parameter
        perturbation = source.integrate_test_mass_motion(times, positions)
        kick = perturbation.as_numpy()["delta_velocity_m_s"][-1]
        expected_y = (
            -2.0
            * G_SI
            * source.rest_mass_kg
            / (impact_parameter * C_SI)
            * (2.0 * source.lorentz_factor**2 - 1.0)
            / (source.lorentz_factor * beta)
        )
        np.testing.assert_allclose(
            kick, [0.0, expected_y, 0.0], rtol=5.0e-6, atol=2.0e-12
        )

    def test_test_mass_perturbation_has_strict_retarded_time_support(self) -> None:
        source = UniformMovingPointMass(
            rest_mass_kg=1.0e20,
            position_at_reference_m=tuple(
                np.mean(self.positions, axis=0) + np.asarray([5.0e9, 4.0e9, 3.0e9])
            ),
            velocity_m_s=(3.0e5, 0.0, 0.0),
            reference_time_s=500.0,
        )
        support_time = np.linspace(0.0, 1000.0, 1001)
        background = np.repeat(
            self.positions[np.newaxis, :, :], len(support_time), axis=0
        )
        motion = source.integrate_test_mass_motion(support_time, background)
        response_time = np.linspace(20.0, 1000.0, 99)
        engine = WeakFieldLinkResponse(orbits=self.orbits, force_backend="cpu")
        result = engine.compute(response_time, source, delta_velocity_m_s=motion)
        self.assertEqual(
            result.metadata["worldline_velocity_source"], "TestMassPerturbation"
        )
        self.assertTrue(np.all(np.isfinite(result.as_numpy()["total"])))

        short_motion = source.integrate_test_mass_motion(
            response_time, background[: len(response_time)]
        )
        with self.assertRaises(ValueError):
            engine.compute(response_time, source, delta_velocity_m_s=short_motion)

    def test_moving_point_mass_degenerate_chord_and_singularity(self) -> None:
        geometry = build_link_geometry(self.orbits, np.asarray([1000.0]))
        emitter = geometry.x_emission_m[0, 0]
        receiver = geometry.x_reception_m[0, 0]
        direction = geometry.direction[0, 0]
        exterior_source = emitter - 2.0e9 * direction
        source = UniformMovingPointMass(
            rest_mass_kg=1.0e25,
            position_at_reference_m=tuple(exterior_source),
            velocity_m_s=(0.0, 0.0, 0.0),
        )
        direct = source.direct_link_signal(geometry)
        self.assertTrue(np.all(np.isfinite(direct)))

        intersecting_source = UniformMovingPointMass(
            rest_mass_kg=1.0e25,
            position_at_reference_m=tuple(0.5 * (emitter + receiver)),
            velocity_m_s=(0.0, 0.0, 0.0),
        )
        with self.assertRaises(ValueError):
            intersecting_source.direct_link_signal(geometry)


if __name__ == "__main__":
    unittest.main()
