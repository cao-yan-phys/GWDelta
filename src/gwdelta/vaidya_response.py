from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.integrate import quad_vec, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import spence

from .fd_response import C_SI, DEFAULT_LINKS


def profile_integrals(model, time, position):
    position = np.asarray(position, dtype=float)
    time = np.asarray(time, dtype=float)
    if time.ndim == 1 and position.ndim > 2 and time.shape[0] == position.shape[0]:
        time = time.reshape((len(time),) + (1,)*(position.ndim-2))
    relative = position-np.asarray(model.origin_m)
    radius = np.linalg.norm(relative, axis=-1)
    if np.any(radius <= 0) or not np.all(np.isfinite(radius)) or not np.all(np.isfinite(time)):
        raise ValueError("Vaidya coordinates must be finite and outside the source")
    reference = position.reshape(-1, 3)[0]
    reference_time = float(time.reshape(-1)[0])
    reference_relative = reference-np.asarray(model.origin_m)
    dx = position-reference
    dr = np.sum(dx*(2*reference_relative+dx), axis=-1)/(radius+np.linalg.norm(reference_relative))
    offset = model._retarded_offset(reference_time, reference)
    elapsed = offset+(time-reference_time)-dr/C_SI
    z = 2*elapsed/model.transition_time_s
    ex = np.exp(-np.abs(z))
    f = np.where(z >= 0, 1., ex)/(1+ex)
    fdot = 2*ex/(model.transition_time_s*(1+ex)**2)
    j = np.maximum(elapsed, 0.)+.5*model.transition_time_s*np.log1p(ex)
    small = np.abs(z) > np.log(2.)
    term = -np.where(small, ex, 0.)
    dilog = term.copy()
    for order in range(2, 49):
        term *= -ex
        contribution = term/order**2
        dilog += contribution
        if np.all(np.abs(contribution) <= 4*np.finfo(float).eps*np.maximum(np.abs(dilog), np.finfo(float).tiny)):
            break
    dilog = np.where(small, dilog, spence(1+ex))
    q = .5*np.maximum(elapsed, 0.)**2+model.transition_time_s**2/4*np.where(
        z <= 0, -dilog, np.pi**2/6+dilog
    )
    return relative/radius[..., None], radius, f, fdot, j, q, z/2


def synchronous_metric(model, time, position):
    from .weak_field import G_SI

    n, radius, f, _fdot, j, q, _argument = profile_integrals(model, time, position)
    mass_scale = G_SI*model.delta_mass_kg/C_SI**2
    eye = np.eye(3)
    nn = n[..., :, None]*n[..., None, :]
    a = C_SI*j/radius**2
    b = C_SI**2*q/radius**3
    h = 2*mass_scale*(a[..., None, None]*(nn-eye)+b[..., None, None]*(eye-3*nn))
    hdot = 2*mass_scale*(
        (C_SI*f/radius**2)[..., None, None]*(nn-eye)
        +(C_SI**2*j/radius**3)[..., None, None]*(eye-3*nn)
    )
    ar = -f/radius**2-2*a/radius
    br = -C_SI*j/radius**3-3*b/radius
    dn = (eye-nn)/radius[..., None, None]
    dnn = np.einsum("...ik,...j->...ijk", dn, n)+np.einsum("...i,...jk->...ijk", n, dn)
    gradient = 2*mass_scale*(
        (ar[..., None, None]*(nn-eye)+br[..., None, None]*(eye-3*nn))[..., None]*n[..., None, None, :]
        +(a-3*b)[..., None, None, None]*dnn
    )
    return h, hdot, gradient


def motion_force(model, time, position, velocity):
    h, hdot, gradient = synchronous_metric(model, time, position)
    velocity = np.asarray(velocity, dtype=float)
    if np.any(np.linalg.norm(velocity, axis=-1) >= C_SI):
        raise ValueError("background velocities must be subluminal")
    first = -np.einsum("...ij,...j->...i", hdot, velocity)
    second = -np.einsum("...ijk,...j,...k->...i", gradient, velocity, velocity)
    second += .5*np.einsum("...jki,...j,...k->...i", gradient, velocity, velocity)
    time_term = .5*np.einsum("...i,...ij,...j->...", velocity, hdot, velocity)/C_SI**2
    return first+second+time_term[..., None]*velocity


def integrate_motion(model, t_s, background_position_m, *, initial_delta_velocity_m_s=None,
                     initial_delta_position_m=None, initial_delta_clock_s=None,
                     background_acceleration_jacobian=None,
                     background_spline=None):
    from .weak_field import G_SI, TestMassPerturbation, _to_numpy, _xp_for

    xp = _xp_for(background_position_m)
    time = np.asarray(_to_numpy(t_s), dtype=float)
    position = np.asarray(_to_numpy(background_position_m), dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    if position.shape[0] != len(time) or position.shape[-1:] != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("background positions must have shape (len(t), ..., 3) and be finite")
    initial_shape = position.shape[1:]
    def initial(value):
        result = np.zeros(initial_shape) if value is None else np.broadcast_to(
            np.asarray(_to_numpy(value), dtype=float), initial_shape
        ).copy()
        if not np.all(np.isfinite(result)):
            raise ValueError("initial perturbations must be finite")
        return result.reshape(-1, 3)
    initial_v = initial(initial_delta_velocity_m_s)
    initial_x = initial(initial_delta_position_m)
    initial_clock = np.zeros(initial_shape[:-1]) if initial_delta_clock_s is None else np.broadcast_to(
        np.asarray(_to_numpy(initial_delta_clock_s), dtype=float), initial_shape[:-1]
    ).copy()
    if not np.all(np.isfinite(initial_clock)):
        raise ValueError("initial clock perturbations must be finite")
    initial_clock = initial_clock.reshape(-1)
    if initial_delta_velocity_m_s is None and initial_delta_position_m is None:
        if np.any(profile_integrals(model, time[0], position[0])[2] > 1e-12):
            raise ValueError("Vaidya motion requires pre-loss support or explicit initial conditions")
    flat = position.reshape(len(time), -1, 3)
    spline = background_spline or CubicSpline(time-time[0], position, axis=0, extrapolate=False)
    span = float(time[-1]-time[0])
    evaluators = []
    for sc in range(flat.shape[1]):
        def trajectory(s, derivative=0, sc=sc):
            return spline(s, derivative).reshape(np.asarray(s).shape+(-1, 3))[..., sc, :]
        r0 = np.linalg.norm(flat[0, sc]-np.asarray(model.origin_m))
        velocity_scale = G_SI*model.delta_mass_kg*span/r0**2
        position_scale = velocity_scale*span
        clock_scale = position_scale/C_SI
        breaks = [0., span]
        def argument(s):
            return float(profile_integrals(model, time[0]+s, trajectory(s))[6])
        a0, a1 = argument(0.), argument(span)
        for target in (-24., -8., -2., 0., 2., 8., 24.):
            if a0 < target < a1:
                breaks.append(brentq(lambda s: argument(s)-target, 0., span,
                                     xtol=max(1e-12, 8*np.finfo(float).eps*span)))
        breaks = np.unique(breaks)
        state = np.concatenate((initial_x[sc]/position_scale, initial_v[sc]/velocity_scale,
                                [initial_clock[sc]/clock_scale]))
        solutions = []
        for lower, upper in zip(breaks[:-1], breaks[1:]):
            def rhs(s, y):
                p = trajectory(s)
                v = trajectory(s, 1)
                force = motion_force(model, time[0]+s, p, v)
                if background_acceleration_jacobian is not None:
                    jacobian = np.asarray(background_acceleration_jacobian(time[0]+s, p, v), dtype=float)
                    if jacobian.shape != (3, 3) or not np.all(np.isfinite(jacobian)):
                        raise ValueError("background acceleration Jacobian must be a finite 3x3 matrix")
                    force += jacobian@(position_scale*y[:3])
                h = synchronous_metric(model, time[0]+s, p)[0]
                sigma = np.sqrt(1-np.dot(v, v)/C_SI**2)
                clock_rate = -(np.dot(v, velocity_scale*y[3:6])+.5*np.dot(v, h@v))/(C_SI**2*sigma)
                return np.concatenate((y[3:6]/span, force/velocity_scale, [clock_rate/clock_scale]))
            sol = solve_ivp(rhs, (lower, upper), state, method="DOP853", rtol=2e-12,
                            atol=2e-14, dense_output=True)
            if not sol.success:
                raise RuntimeError(sol.message)
            solutions.append(sol)
            state = sol.y[:, -1]
        def evaluate(query, solutions=solutions, breaks=breaks,
                     velocity_scale=velocity_scale, position_scale=position_scale, clock_scale=clock_scale):
            s = np.asarray(query, dtype=float)-time[0]
            if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > span):
                raise ValueError("motion queries must lie within the integrated support")
            segments = np.clip(np.searchsorted(breaks, s, side="right")-1, 0, len(solutions)-1)
            value = np.empty(s.shape+(7,))
            for segment in np.unique(segments):
                mask = segments == segment
                value[mask] = solutions[int(segment)].sol(s[mask]).T
            return value[..., :3]*position_scale, value[..., 3:6]*velocity_scale, value[..., 6]*clock_scale
        evaluators.append(evaluate)
    def evaluate_all(query, component):
        query = np.asarray(query, dtype=float)
        values = [evaluate(query)[component] for evaluate in evaluators]
        if component == 2:
            return np.stack(values, axis=-1).reshape(query.shape+initial_shape[:-1])
        return np.stack(values, axis=-2).reshape(query.shape+initial_shape)
    velocities = spline(time-time[0], 1)
    return TestMassPerturbation(
        t=xp.asarray(time), acceleration_m_s2=xp.asarray(motion_force(model, time, position, velocities)),
        delta_velocity_m_s=xp.asarray(evaluate_all(time, 1)),
        velocity_evaluator=lambda query: evaluate_all(query, 1),
        position_evaluator=lambda query: evaluate_all(query, 0),
        clock_evaluator=lambda query: evaluate_all(query, 2),
        background_velocity_evaluator=lambda query: spline(np.asarray(query)-time[0], 1),
        source_model=model, coordinate_gauge="synchronous",
        background_feedback=background_acceleration_jacobian is not None,
    )


def refine_geometry(orbits, geometry):
    from .weak_field import _to_numpy

    tbase = np.asarray(_to_numpy(orbits.t_base), dtype=float)
    xbase = np.asarray(_to_numpy(orbits.x_base), dtype=float)
    spline = CubicSpline(tbase-tbase[0], xbase, axis=0, extrapolate=True)
    receivers = np.array([int(str(link)[0])-1 for link in DEFAULT_LINKS])
    emitters = np.array([int(str(link)[1])-1 for link in DEFAULT_LINKS])
    reception = geometry.t_reception_s
    tr = reception[None, :]
    xr = np.moveaxis(spline(reception-tbase[0])[:, receivers], 0, 1)
    te = geometry.t_emission_s.copy()
    for _ in range(20):
        xe = np.stack([spline(te[link]-tbase[0])[:, emitter] for link, emitter in enumerate(emitters)])
        next_te = tr-np.linalg.norm(xr-xe, axis=-1)/C_SI
        if np.max(np.abs(next_te-te)) <= 8*np.finfo(float).eps*max(1., np.max(np.abs(reception))):
            te = next_te
            break
        te = next_te
    else:
        allowed = max(2*float(np.max(geometry.light_time_s)), float(np.median(np.diff(tbase))))
        lower_limit = float(tbase[0]-allowed)
        for link, sample in np.ndindex(te.shape):
            reception_time = float(reception[sample])
            receiver_position = xr[link, sample]
            emitter = emitters[link]
            def equation(emission_time):
                emitter_position = spline(emission_time-tbase[0])[emitter]
                return reception_time-emission_time-np.linalg.norm(receiver_position-emitter_position)/C_SI
            if equation(lower_limit) < 0:
                raise ValueError("retarded Vaidya link requires unavailable orbit support")
            te[link, sample] = brentq(equation, lower_limit, reception_time, xtol=1e-12)
    xe = np.stack([spline(te[link]-tbase[0])[:, emitter] for link, emitter in enumerate(emitters)])
    ve = np.stack([spline(te[link]-tbase[0], 1)[:, emitter] for link, emitter in enumerate(emitters)])
    vr = np.moveaxis(spline(reception-tbase[0], 1)[:, receivers], 0, 1)
    chord = xr-xe
    length = np.linalg.norm(chord, axis=-1)
    return replace(geometry, t_emission_s=te, x_reception_m=xr, x_emission_m=xe,
                   v_reception_m_s=vr, v_emission_m_s=ve, direction=chord/length[..., None],
                   light_time_s=length/C_SI, chord_length_m=length,
                   max_null_mismatch=0.,
                   max_emission_extrapolation_s=max(0., float(tbase[0]-np.min(te))))


def link_response(model, geometry, motion, orbits, *, backend):
    from .weak_field import G_SI, LinkSignalResult, _to_numpy

    if motion.source_model != model or motion.coordinate_gauge != "synchronous":
        raise ValueError("complete Vaidya response requires motion in the synchronous gauge of this model")
    if not callable(motion.position_evaluator) or not callable(motion.velocity_evaluator) or not callable(motion.clock_evaluator):
        raise ValueError("complete Vaidya response requires dense position and velocity solutions")
    tbase = np.asarray(_to_numpy(orbits.t_base), dtype=float)
    spline = CubicSpline(tbase-tbase[0], np.asarray(_to_numpy(orbits.x_base), dtype=float), axis=0)
    receivers = np.array([int(str(link)[0])-1 for link in geometry.links])
    emitters = np.array([int(str(link)[1])-1 for link in geometry.links])
    dxr = motion.position_evaluator(geometry.t_reception_s)
    dvr = motion.velocity_evaluator(geometry.t_reception_s)
    clock_r = motion.clock_evaluator(geometry.t_reception_s)
    if dxr.shape != (len(geometry.t_reception_s), 3, 3) or dvr.shape != dxr.shape or clock_r.shape != (len(geometry.t_reception_s), 3):
        raise ValueError("Vaidya motion must cover three spacecraft with dense positions, velocities, and clocks")
    if not all(np.all(np.isfinite(values)) for values in (dxr, dvr, clock_r)):
        raise ValueError("Vaidya motion values must be finite")
    direct = np.empty(geometry.t_emission_s.shape)
    worldline = np.empty_like(direct)
    for link, sample in np.ndindex(direct.shape):
        receiver, emitter = receivers[link], emitters[link]
        tr = float(geometry.t_reception_s[sample])
        te = float(geometry.t_emission_s[link, sample])
        xr, xe = geometry.x_reception_m[link, sample], geometry.x_emission_m[link, sample]
        vr, ve = geometry.v_reception_m_s[link, sample], geometry.v_emission_m_s[link, sample]
        ar, ae = spline(tr-tbase[0], 2)[receiver], spline(te-tbase[0], 2)[emitter]
        k = geometry.direction[link, sample]
        length = geometry.chord_length_m[link, sample]
        relative = xe-np.asarray(model.origin_m)
        chord = xr-xe
        closest = np.clip(-np.dot(relative, chord)/length**2, 0., 1.)
        tolerance = 128*np.finfo(float).eps*max(np.linalg.norm(relative), length)
        if np.linalg.norm(relative+closest*chord) <= tolerance:
            raise ValueError("photon chord intersects the Vaidya source origin")
        light_time = length/C_SI
        alpha_e, alpha_r = 1-np.dot(k, ve)/C_SI, 1-np.dot(k, vr)/C_SI
        if alpha_e <= 0 or alpha_r <= 0 or max(np.linalg.norm(ve), np.linalg.norm(vr)) >= C_SI:
            raise ValueError("Vaidya link endpoints must be subluminal")
        emission_rate = alpha_r/alpha_e
        separation_rate = vr-ve*emission_rate
        length_rate = np.dot(k, separation_rate)
        kdot = (separation_rate-k*length_rate)/length
        alpha_e_rate = -(np.dot(kdot, ve)+np.dot(k, ae)*emission_rate)/C_SI
        position_e = motion.position_evaluator(np.array([te]))[0, emitter]
        velocity_e = motion.velocity_evaluator(np.array([te]))[0, emitter]
        position_r, velocity_r = dxr[sample, receiver], dvr[sample, receiver]
        q_position = np.dot(k, position_r-position_e)/C_SI
        q_position_rate = (np.dot(kdot, position_r-position_e)
                           +np.dot(k, velocity_r-velocity_e*emission_rate))/C_SI
        gamma_e2 = 1/(1-np.dot(ve, ve)/C_SI**2)
        gamma_r2 = 1/(1-np.dot(vr, vr)/C_SI**2)
        emitter_clock_rate = -gamma_e2*np.dot(ve, ae)/C_SI**2
        receiver_clock_rate = -gamma_r2*np.dot(vr, ar)/C_SI**2
        alpha_r_rate = -(np.dot(kdot, vr)+np.dot(k, ar))/C_SI
        background_frequency_rate = emitter_clock_rate*emission_rate-receiver_clock_rate
        background_frequency_rate += alpha_r_rate/alpha_r-alpha_e_rate/alpha_e
        def path(z):
            return te+light_time*z, xe+(xr-xe)*z
        def argument(z):
            tp, xp = path(z)
            return float(profile_integrals(model, tp, xp)[6])
        breaks = [0., 1.]
        a0, a1 = argument(0.), argument(1.)
        for target in (-24., -8., -2., 0., 2., 8., 24.):
            if a0 < target < a1:
                breaks.append(brentq(lambda z: argument(z)-target, 0., 1., xtol=1e-14))
        breaks = np.unique(breaks)
        time_scale = max(model.transition_time_s, light_time,
                         abs(model.transition_time_s*a0), abs(model.transition_time_s*a1))
        radius = np.linalg.norm(xe-np.asarray(model.origin_m))
        metric_scale = max(np.finfo(float).tiny,
                           2*G_SI*model.delta_mass_kg/C_SI*max(1., abs(tr-te))/radius**2)
        def integrand(z):
            tp, xp = path(z)
            h, hdot, grad = synchronous_metric(model, tp, xp)
            xp_rate = ve*emission_rate*(1-z)+vr*z
            tp_rate = emission_rate*(1-z)+z
            derivative = hdot*tp_rate+np.einsum("ijk,k->ij", grad, xp_rate)
            contraction = np.dot(k, h@k)
            contraction_rate = np.dot(k, derivative@k)+2*np.dot(kdot, h@k)
            return np.array([contraction/time_scale, contraction_rate])*light_time/(2*metric_scale)
        integral = np.zeros(2)
        for a, b in zip(breaks[:-1], breaks[1:]):
            value, error, info = quad_vec(integrand, a, b, epsabs=2e-13, epsrel=2e-12,
                                         norm="max", full_output=True)
            if not info.success:
                raise RuntimeError("Vaidya time-transfer integral failed to converge")
            integral += value
        delay = metric_scale*integral[0]*time_scale
        delay_rate = metric_scale*integral[1]
        delay_rate += length_rate/length*delay
        he = synchronous_metric(model, te, xe)[0]
        hr = synchronous_metric(model, tr, xr)[0]
        clock_metric = -.5*gamma_e2*np.dot(ve, he@ve)/C_SI**2+.5*gamma_r2*np.dot(vr, hr@vr)/C_SI**2
        clock_motion = -gamma_e2*np.dot(ve, velocity_e)/C_SI**2+gamma_r2*np.dot(vr, velocity_r)/C_SI**2
        direct[link, sample] = clock_metric-delay_rate/alpha_r+delay*(alpha_e_rate/alpha_r-emitter_clock_rate)/alpha_e
        worldline[link, sample] = clock_motion-q_position_rate/alpha_r+q_position*(alpha_e_rate/alpha_r-emitter_clock_rate)/alpha_e
        worldline[link, sample] -= background_frequency_rate*clock_r[sample, receiver]*np.sqrt(gamma_r2)
    total = direct+worldline
    return LinkSignalResult(
        t=geometry.t_reception_s.copy(), links=geometry.links,
        direct=backend.xp.asarray(direct), worldline=backend.xp.asarray(worldline),
        total=backend.xp.asarray(total), geometry=geometry,
        metadata={"backend": backend.name, "response_order": "linear metric, moving-endpoint time transfer",
                  "frequency_normalization": "unperturbed_received_carrier",
                  "time_matching": "receiver_proper_time", "integration_backend": "scipy_cpu",
                  "response_gauge": "synchronous", "direct_evaluation": "vaidya_time_transfer",
                  "worldline_velocity_included": True, "worldline_velocity_source": "TestMassPerturbation",
                  "link_order": list(geometry.links), "max_null_mismatch": geometry.max_null_mismatch,
                  "max_emission_extrapolation_s": geometry.max_emission_extrapolation_s},
    )
