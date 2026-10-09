from pathlib import Path
import json
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gwdelta.orbits import (
    AU_SI,
    SIDEREAL_YEAR_S,
    build_lisa_simple_orbit_arrays,
    build_taiji_simple_orbit_arrays,
    build_tianqin_simple_orbit_arrays,
)


OUTPUT = PROJECT_ROOT / "docs" / "interactive" / "constellation_orbits.html"
SPACECRAFT_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
SPACECRAFT_NAMES = ("1", "2", "3")
DISPLAY_SPACECRAFT_ORDER = (1, 0, 2)
TRAJECTORY_SAMPLES = 721
SLIDER_SAMPLES = 121
DISPLAY_ARM_SCALES = {"LISA": 1.0, "Taiji": 1.0, "TianQin": 20.0}
GUIDING_CENTER_COLORS = {"LISA": "#9a5b2e", "Taiji": "#5d7e4a", "TianQin": "#496a95"}
NORMAL_ARROW_LENGTH_AU = 0.18
NORMAL_ARROW_HEAD_AU = 0.05
NORMAL_ARROW_LENGTH_LOCAL = 0.70
NORMAL_ARROW_HEAD_LOCAL = 0.09


def orbit_models():
    duration_s = SIDEREAL_YEAR_S
    orbit_dt_s = duration_s / (TRAJECTORY_SAMPLES - 1)
    return (
        ("LISA", build_lisa_simple_orbit_arrays(duration=duration_s, orbit_dt=orbit_dt_s)),
        ("Taiji", build_taiji_simple_orbit_arrays(duration=duration_s, orbit_dt=orbit_dt_s)),
        ("TianQin", build_tianqin_simple_orbit_arrays(duration=duration_s, orbit_dt=orbit_dt_s)),
    )


def axis(title, value_range):
    return {
        "title": title,
        "range": value_range,
        "showbackground": False,
        "showgrid": True,
        "gridcolor": "#d7dde3",
        "showline": True,
        "linecolor": "#82909b",
        "zeroline": False,
        "ticks": "outside",
    }


def trajectory_scene():
    return {
        "xaxis": axis("<i>x</i>' [AU]", [-1.25, 1.25]),
        "yaxis": axis("<i>y</i>' [AU]", [-1.25, 1.25]),
        "zaxis": axis("<i>z</i>' [AU]", [-1.25, 1.25]),
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.15, "y": 1.15, "z": 1.5}},
    }


def local_scene():
    return {
        "xaxis": axis("Δ<i>x</i>' / <i>L</i>", [-0.72, 0.72]),
        "yaxis": axis("Δ<i>y</i>' / <i>L</i>", [-0.72, 0.72]),
        "zaxis": axis("Δ<i>z</i>' / <i>L</i>", [-0.72, 0.72]),
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.35, "y": 1.35, "z": 0.9}},
    }


def corotating_local_scene():
    return {
        "xaxis": axis("Δ<i>x</i>'' / <i>L</i>", [-0.72, 0.72]),
        "yaxis": axis("Δ<i>y</i>'' / <i>L</i>", [-0.72, 0.72]),
        "zaxis": axis("Δ<i>z</i>'' / <i>L</i>", [-0.72, 0.72]),
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.35, "y": 1.35, "z": 0.9}},
    }


def display_label(name):
    scale = DISPLAY_ARM_SCALES[name]
    if scale == 1.0:
        return name
    return f"{name} (<i>L</i> × {scale:g})"


def guiding_centers_au(arrays):
    return arrays.x.mean(axis=1) / AU_SI


def display_spacecraft_positions(positions):
    return np.asarray(positions)[list(DISPLAY_SPACECRAFT_ORDER)]


def displayed_constellation_au(name, arrays, time_index):
    center = arrays.x[time_index].mean(axis=0)
    relative = arrays.x[time_index] - center
    positions = center + DISPLAY_ARM_SCALES[name] * relative
    return display_spacecraft_positions(positions) / AU_SI


def spacecraft_order_normal(arrays, time_index):
    positions = display_spacecraft_positions(arrays.x[time_index])
    normal = -np.cross(positions[1] - positions[0], positions[2] - positions[0])
    return normal / np.linalg.norm(normal)


def spacecraft_order_arrow_au(arrays, time_index):
    tail = arrays.x[time_index].mean(axis=0) / AU_SI
    direction = NORMAL_ARROW_LENGTH_AU * spacecraft_order_normal(arrays, time_index)
    return tail, direction


def arrow_head_direction(direction, head_length):
    return head_length * direction / np.linalg.norm(direction)


def add_guiding_center_trajectory(fig, name, arrays):
    centers_au = guiding_centers_au(arrays)
    fig.add_trace(
        go.Scatter3d(
            x=centers_au[:, 0],
            y=centers_au[:, 1],
            z=centers_au[:, 2],
            mode="lines",
            line={"color": GUIDING_CENTER_COLORS[name], "width": 2},
            opacity=0.35,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def normalized_configuration(arrays, time_index):
    relative = arrays.x[time_index] - arrays.x[time_index].mean(axis=0)
    return display_spacecraft_positions(relative) / arrays.armlength


def rotation_minimizing_bases(arrays):
    normals = np.asarray([spacecraft_order_normal(arrays, index) for index in range(len(arrays.t))])
    bases = np.empty((len(normals), 3, 3), dtype=float)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normals[0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = reference - np.dot(reference, normals[0]) * normals[0]
    e1 /= np.linalg.norm(e1)
    bases[0] = np.stack((e1, np.cross(normals[0], e1), normals[0]))
    for index in range(1, len(normals)):
        previous = normals[index - 1]
        current = normals[index]
        axis_vector = np.cross(previous, current)
        sine = np.linalg.norm(axis_vector)
        cosine = np.clip(np.dot(previous, current), -1.0, 1.0)
        e1 = bases[index - 1, 0]
        if sine > 1.0e-14:
            axis_vector /= sine
            e1 = (
                cosine * e1
                + sine * np.cross(axis_vector, e1)
                + (1.0 - cosine) * np.dot(axis_vector, e1) * axis_vector
            )
        e1 -= np.dot(e1, current) * current
        e1 /= np.linalg.norm(e1)
        bases[index] = np.stack((e1, np.cross(current, e1), current))
    return bases


def corotating_normalized_configuration(arrays, bases, time_index):
    coordinates = normalized_configuration(arrays, time_index)
    return coordinates @ bases[time_index].T


def add_current_constellation(fig, name, arrays):
    positions_au = displayed_constellation_au(name, arrays, 0)
    triangle = np.vstack((positions_au, positions_au[0]))
    fig.add_trace(
        go.Scatter3d(
            x=triangle[:, 0],
            y=triangle[:, 1],
            z=triangle[:, 2],
            mode="lines",
            line={"color": "#000000", "width": 6},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=positions_au[:, 0],
            y=positions_au[:, 1],
            z=positions_au[:, 2],
            mode="markers",
            text=SPACECRAFT_NAMES,
            marker={"color": SPACECRAFT_COLORS, "size": 3.5, "line": {"color": "#ffffff", "width": 0.6}},
            hovertemplate=(
                f"{display_label(name)}, %{{text}}<br>"
                "<i>x</i>' = %{x:.5f} AU<br><i>y</i>' = %{y:.5f} AU<br><i>z</i>' = %{z:.5f} AU<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    tail, direction = spacecraft_order_arrow_au(arrays, 0)
    head = tail + direction
    head_direction = arrow_head_direction(direction, NORMAL_ARROW_HEAD_AU)
    cone_tail = head
    fig.add_trace(
        go.Scatter3d(
            x=[tail[0], head[0]],
            y=[tail[1], head[1]],
            z=[tail[2], head[2]],
            mode="lines",
            line={"color": "#707070", "width": 6},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Cone(
            x=[cone_tail[0]],
            y=[cone_tail[1]],
            z=[cone_tail[2]],
            u=[head_direction[0]],
            v=[head_direction[1]],
            w=[head_direction[2]],
            anchor="tail",
            colorscale=[[0.0, "#707070"], [1.0, "#707070"]],
            showscale=False,
            sizemode="absolute",
            sizeref=NORMAL_ARROW_HEAD_AU,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def add_local_configuration(fig, name, arrays, visible, *, row, col, bases=None):
    coordinates = (
        normalized_configuration(arrays, 0)
        if bases is None
        else corotating_normalized_configuration(arrays, bases, 0)
    )
    suffix = "'" if bases is None else "''"
    direction = (
        NORMAL_ARROW_LENGTH_LOCAL * spacecraft_order_normal(arrays, 0)
        if bases is None
        else np.array([0.0, 0.0, NORMAL_ARROW_LENGTH_LOCAL])
    )
    triangle = np.vstack((coordinates, coordinates[0]))
    fig.add_trace(
        go.Scatter3d(
            x=triangle[:, 0],
            y=triangle[:, 1],
            z=triangle[:, 2],
            mode="lines",
            line={"color": "#000000", "width": 5},
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="markers+text",
            text=SPACECRAFT_NAMES,
            textposition="top center",
            textfont={"size": 14, "color": "#18212b"},
            marker={"color": SPACECRAFT_COLORS, "size": 7, "line": {"color": "#ffffff", "width": 0.8}},
            hovertemplate=(
                f"{name}, %{{text}}, <i>t</i> = 0 d<br>"
                f"Δ<i>x</i>{suffix} / <i>L</i> = %{{x:.4f}}<br>Δ<i>y</i>{suffix} / <i>L</i> = %{{y:.4f}}<br>Δ<i>z</i>{suffix} / <i>L</i> = %{{z:.4f}}<extra></extra>"
            ),
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=col,
    )
    head_direction = arrow_head_direction(direction, NORMAL_ARROW_HEAD_LOCAL)
    cone_tail = direction
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, direction[0]],
            y=[0.0, direction[1]],
            z=[0.0, direction[2]],
            mode="lines",
            line={"color": "#707070", "width": 6},
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Cone(
            x=[cone_tail[0]],
            y=[cone_tail[1]],
            z=[cone_tail[2]],
            u=[head_direction[0]],
            v=[head_direction[1]],
            w=[head_direction[2]],
            anchor="tail",
            colorscale=[[0.0, "#707070"], [1.0, "#707070"]],
            showscale=False,
            sizemode="absolute",
            sizeref=NORMAL_ARROW_HEAD_LOCAL,
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=col,
    )


def local_configuration_frame_data(name, arrays, coordinates, time_index, *, bases=None):
    suffix = "'" if bases is None else "''"
    direction = (
        NORMAL_ARROW_LENGTH_LOCAL * spacecraft_order_normal(arrays, time_index)
        if bases is None
        else np.array([0.0, 0.0, NORMAL_ARROW_LENGTH_LOCAL])
    )
    time_days = time_index * SIDEREAL_YEAR_S / (TRAJECTORY_SAMPLES - 1) / 86400.0
    triangle = np.vstack((coordinates, coordinates[0]))
    head_direction = arrow_head_direction(direction, NORMAL_ARROW_HEAD_LOCAL)
    return (
        go.Scatter3d(x=triangle[:, 0], y=triangle[:, 1], z=triangle[:, 2]),
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            hovertemplate=(
                f"{name}, %{{text}}, <i>t</i> = {time_days:.1f} d<br>"
                f"Δ<i>x</i>{suffix} / <i>L</i> = %{{x:.4f}}<br>Δ<i>y</i>{suffix} / <i>L</i> = %{{y:.4f}}<br>Δ<i>z</i>{suffix} / <i>L</i> = %{{z:.4f}}<extra></extra>"
            ),
        ),
        go.Scatter3d(x=[0.0, direction[0]], y=[0.0, direction[1]], z=[0.0, direction[2]]),
        go.Cone(
            x=[direction[0]],
            y=[direction[1]],
            z=[direction[2]],
            u=[head_direction[0]],
            v=[head_direction[1]],
            w=[head_direction[2]],
        ),
    )


def toggle_script(groups):
    return """\
const plot = document.getElementById('{plot_id}');
const groups = __GROUPS__;
const panel = document.createElement('div');
panel.style.cssText = 'position:absolute;left:16px;top:16px;z-index:10;display:grid;gap:4px;';
const host = plot.parentElement;
host.style.position = 'relative';
host.appendChild(panel);
for (const [label, traces] of Object.entries(groups)) {
  const button = document.createElement('button');
  let shown = true;
  button.innerHTML = label;
  button.style.cssText = 'background:rgba(255,255,255,0.88);border:1px solid #b8c2cc;border-radius:2px;color:#18212b;cursor:pointer;font:12px Open Sans,Arial,sans-serif;padding:4px 8px;text-align:left;';
  button.addEventListener('click', () => {
    shown = !shown;
    button.style.opacity = shown ? '1' : '0.42';
    Plotly.restyle(plot, {visible: shown}, traces);
  });
  panel.appendChild(button);
}
plot.on('plotly_buttonclicked', (event) => {
  if (event.button?.name !== 'align') return;
  const camera = JSON.parse(JSON.stringify(plot.layout.scene.camera));
  Plotly.relayout(plot, {'scene2.camera': camera});
});
const cameraFor = (sceneName) => {
  const scene = plot._fullLayout[sceneName]?._scene;
  return scene && scene.getCamera ? scene.getCamera() : plot.layout[sceneName].camera;
};
const rotateAboutVertical = (camera, angle) => {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const rotate = ({x, y, z}) => ({x: cosine * x - sine * y, y: sine * x + cosine * y, z});
  return {...camera, eye: rotate(camera.eye), center: rotate(camera.center), up: rotate(camera.up)};
};
let horizontalRotation = false;
let drag = null;
let animationFrame = null;
const setHorizontalRotationButton = (active) => {
  horizontalRotation = active;
  horizontalButton.classList.toggle('active', active);
};
const turntableButton = Array.from(plot.querySelectorAll('.modebar-btn')).find(
  (button) => button.getAttribute('data-title') === 'Turntable rotation',
);
const horizontalButton = turntableButton.cloneNode(true);
turntableButton.replaceWith(horizontalButton);
horizontalButton.addEventListener('click', () => {
  setHorizontalRotationButton(true);
  Plotly.relayout(plot, {'scene.dragmode': false, 'scene2.dragmode': false, 'scene3.dragmode': false});
});
for (const title of ['Pan', 'Orbital rotation', 'Reset camera to default']) {
  const button = Array.from(plot.querySelectorAll('.modebar-btn')).find(
    (item) => item.getAttribute('data-title') === title,
  );
  if (button) button.addEventListener('click', () => setHorizontalRotationButton(false));
}
const beginHorizontalDrag = (event) => {
  if (!horizontalRotation || event.button !== 0) return;
  if (!plot.contains(event.target)) return;
  const bounds = plot.getBoundingClientRect();
  const x = (event.clientX - bounds.left) / bounds.width;
  const y = 1.0 - (event.clientY - bounds.top) / bounds.height;
  const sceneName = ['scene', 'scene2', 'scene3'].find((name) => {
    const domain = plot._fullLayout[name]?.domain;
    return domain && x >= domain.x[0] && x <= domain.x[1] && y >= domain.y[0] && y <= domain.y[1];
  });
  if (!sceneName) return;
  drag = {
    camera: cameraFor(sceneName),
    sceneName,
    startX: event.clientX,
    width: bounds.width * (sceneName === 'scene3' ? 0.50 : 0.47),
  };
  event.preventDefault();
  event.stopImmediatePropagation();
};
const continueHorizontalDrag = (event) => {
  if (!drag) return;
  const current = drag;
  current.offsetX = event.clientX - current.startX;
  if (animationFrame !== null) return;
  animationFrame = window.requestAnimationFrame(() => {
    const angle = -2.0 * Math.PI * current.offsetX / current.width;
    Plotly.relayout(plot, {[`${current.sceneName}.camera`]: rotateAboutVertical(current.camera, angle)});
    animationFrame = null;
  });
  event.preventDefault();
  event.stopImmediatePropagation();
};
document.addEventListener('mousedown', beginHorizontalDrag, true);
document.addEventListener('mousemove', continueHorizontalDrag, true);
document.addEventListener('mouseup', () => {
  drag = null;
}, true);
""".replace("__GROUPS__", json.dumps(groups))


def build_figure():
    models = orbit_models()
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
        row_heights=[0.52, 0.48],
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={"color": "#d99a00", "size": 7},
            hovertemplate="Sun<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    tianqin_arrays = dict(models)["TianQin"]
    earth_positions_au = tianqin_arrays.x.mean(axis=1) / AU_SI
    fig.add_trace(
        go.Scatter3d(
            x=[earth_positions_au[0, 0]],
            y=[earth_positions_au[0, 1]],
            z=[earth_positions_au[0, 2]],
            mode="markers",
            marker={"color": "#000000", "size": 3.5},
            hovertemplate="Earth, <i>t</i> = 0 d<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    guiding_center_trace_indices = {}
    for name, arrays in models:
        guiding_center_trace_indices[name] = len(fig.data)
        add_guiding_center_trajectory(fig, name, arrays)

    current_trace_indices = []
    for name, arrays in models:
        before = len(fig.data)
        add_current_constellation(fig, name, arrays)
        current_trace_indices.append(tuple(range(before, len(fig.data))))

    earth_trace_index = 1
    local_prime_trace_indices = []
    corotating_trace_indices = []
    for index, (name, arrays) in enumerate(models):
        before = len(fig.data)
        add_local_configuration(fig, name, arrays, visible=index == 0, row=2, col=1)
        local_prime_trace_indices.append(tuple(range(before, len(fig.data))))
        before = len(fig.data)
        add_local_configuration(
            fig,
            name,
            arrays,
            visible=index == 0,
            row=2,
            col=2,
            bases=rotation_minimizing_bases(arrays),
        )
        corotating_trace_indices.append(tuple(range(before, len(fig.data))))

    local_trace_indices = [
        trace_index
        for groups in (local_prime_trace_indices, corotating_trace_indices)
        for group in groups
        for trace_index in group
    ]
    buttons = []
    for index, (name, _arrays) in enumerate(models):
        visible = [False] * len(local_trace_indices)
        for trace_index in (*local_prime_trace_indices[index], *corotating_trace_indices[index]):
            visible[local_trace_indices.index(trace_index)] = True
        buttons.append(
            {
                "label": name,
                "method": "restyle",
                "args": [{"visible": visible}, local_trace_indices],
            }
        )

    slider_indices = np.linspace(0, TRAJECTORY_SAMPLES - 1, SLIDER_SAMPLES, dtype=int)
    frame_trace_indices = [earth_trace_index]
    frame_trace_indices.extend(trace_index for group in current_trace_indices for trace_index in group)
    frame_trace_indices.extend(trace_index for group in local_prime_trace_indices for trace_index in group)
    frame_trace_indices.extend(trace_index for group in corotating_trace_indices for trace_index in group)
    corotating_bases = {name: rotation_minimizing_bases(arrays) for name, arrays in models}
    frames = []
    for time_index in slider_indices:
        frame_data = [
            go.Scatter3d(
                x=[earth_positions_au[time_index, 0]],
                y=[earth_positions_au[time_index, 1]],
                z=[earth_positions_au[time_index, 2]],
                hovertemplate=f"Earth, <i>t</i> = {time_index * SIDEREAL_YEAR_S / (TRAJECTORY_SAMPLES - 1) / 86400.0:.1f} d<extra></extra>",
            )
        ]
        for name, arrays in models:
            positions_au = displayed_constellation_au(name, arrays, time_index)
            triangle = np.vstack((positions_au, positions_au[0]))
            tail, direction = spacecraft_order_arrow_au(arrays, time_index)
            head = tail + direction
            head_direction = arrow_head_direction(direction, NORMAL_ARROW_HEAD_AU)
            cone_tail = head
            frame_data.extend(
                (
                    go.Scatter3d(x=triangle[:, 0], y=triangle[:, 1], z=triangle[:, 2]),
                    go.Scatter3d(
                        x=positions_au[:, 0],
                        y=positions_au[:, 1],
                        z=positions_au[:, 2],
                        hovertemplate=(
                            f"{display_label(name)}, <i>t</i> = {time_index * SIDEREAL_YEAR_S / (TRAJECTORY_SAMPLES - 1) / 86400.0:.1f} d<br>"
                            "<i>x</i>' = %{x:.5f} AU<br><i>y</i>' = %{y:.5f} AU<br><i>z</i>' = %{z:.5f} AU<extra></extra>"
                        ),
                    ),
                    go.Scatter3d(
                        x=[tail[0], head[0]],
                        y=[tail[1], head[1]],
                        z=[tail[2], head[2]],
                    ),
                    go.Cone(
                        x=[cone_tail[0]],
                        y=[cone_tail[1]],
                        z=[cone_tail[2]],
                        u=[head_direction[0]],
                        v=[head_direction[1]],
                        w=[head_direction[2]],
                    ),
                )
            )
        for name, arrays in models:
            coordinates = normalized_configuration(arrays, time_index)
            frame_data.extend(local_configuration_frame_data(name, arrays, coordinates, time_index))
        for name, arrays in models:
            coordinates = corotating_normalized_configuration(arrays, corotating_bases[name], time_index)
            frame_data.extend(
                local_configuration_frame_data(name, arrays, coordinates, time_index, bases=corotating_bases[name])
            )
        frames.append(
            go.Frame(
                name=str(time_index),
                data=frame_data,
                traces=frame_trace_indices,
            )
        )

    slider_steps = [
        {
            "label": f"{time_index * SIDEREAL_YEAR_S / (TRAJECTORY_SAMPLES - 1) / 86400.0:.0f}",
            "method": "animate",
            "args": [
                [str(time_index)],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
            ],
        }
        for time_index in slider_indices
    ]

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=1260,
        margin={"l": 12, "r": 12, "t": 62, "b": 12},
        showlegend=False,
        updatemenus=[
            {
                "buttons": buttons,
                "active": 0,
                "direction": "down",
                "showactive": True,
                "x": 0.89,
                "xanchor": "center",
                "y": 0.48,
                "yanchor": "bottom",
            },
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.01,
                "xanchor": "left",
                "y": 0.48,
                "yanchor": "top",
                "pad": {"r": 5, "t": 10},
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {"duration": 70, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.83,
                "xanchor": "right",
                "y": 0.48,
                "yanchor": "bottom",
                "pad": {"r": 4, "t": 0, "b": 0},
                "buttons": [
                    {
                        "label": "align",
                        "name": "align",
                        "method": "skip",
                    }
                ],
            },
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "<i>t</i> = ", "suffix": " d"},
                "len": 0.64,
                "x": 0.14,
                "xanchor": "left",
                "y": 0.48,
                "yanchor": "top",
                "pad": {"t": 10, "b": 0},
                "steps": slider_steps,
            }
        ],
        scene=trajectory_scene(),
        scene2=local_scene(),
        scene3=corotating_local_scene(),
    )
    fig.frames = frames
    toggle_groups = {"Sun": [0], "Earth": [earth_trace_index]}
    for index, (name, _arrays) in enumerate(models):
        toggle_groups[display_label(name)] = [guiding_center_trace_indices[name], *current_trace_indices[index]]
    return fig, toggle_groups


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure, toggle_groups = build_figure()
    figure.write_html(
        OUTPUT,
        include_plotlyjs=True,
        auto_play=False,
        post_script=toggle_script(toggle_groups),
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["zoom3d", "resetCameraLastSave3d"],
        },
    )
