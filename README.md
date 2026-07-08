# GWDelta

<p align="center">
  <img src="docs/figures/logo.png" alt="GWDelta logo" width="300">
</p>

GWDelta is a toolkit for fast response calculations for space-based gravitational-wave detectors, focusing on LISA-like triangular constellations.

The response code can run on CPU or on a CUDA-enabled modified `fastlisaresponse` backend with `lisatools` through `force_backend="cuda12x"`. GWDelta does not bundle waveform generators; example waveform helpers kept outside `src/gwdelta` may use CuPy or Numba CUDA when available.

## Dependencies

The time-domain response path is tested with the modified `fastlisaresponse` fork [`cao-yan-phys/lisa-on-gpu`](https://github.com/cao-yan-phys/lisa-on-gpu). The fork contains the FastLISAResponse changes used by GWDelta. `fastlisaresponse` provides the `pyResponseTDI` engine; `lisatools` provides the orbit interface expected by that engine and the built-in LISA `esa` / `equal-armlength` orbit classes.

## Example 1

The example below compares a precessing quasi-circular SMBHB waveform generated with `SEOBNRv5PHM` and perturbative displacement memory (including all $l=2$ modes computed perturbatively) against three Taiji response calculations:

- second-generation $A,E$ channels with a realistic Taiji orbit;
- second-generation $A,E$ channels with a static equal-arm orbit;
- an analytic static equal-arm frequency-domain response.

![Taiji TDI response comparison](docs/figures/taiji_static_tdi2_memory_demo.png)



<img src="docs/figures/taiji_ae_time_frequency.png" alt="Taiji A/E time-frequency map" width="450">

## Example 2

The example below compares a one-year eccentric nearly-equal-mass compact binary waveform generated with analytic kludge (AK) and PN models against two Taiji response calculations:

- second-generation $A,E$ channels with a realistic Taiji orbit;
- second-generation $A,E$ channels with a simple equal-arm orbit.

Binary masses: $m_1=50\,M_\odot$, $m_2=30\,M_\odot$; symmetric mass ratio: $\nu=0.234375$; luminosity distance: $100\,\mathrm{Mpc}$; eccentricity: $e_t=0.1$; sampling: `dt=8 s`, `years=1`; frequency markers: $\text{f22\_start}=5.000\,\mathrm{mHz}$ and $\text{f22\_end}\simeq 5.025\,\mathrm{mHz}$.

In the PN model, the initial 1PN QK parameters are aligned to the initial radial mean motion, eccentricity, eccentric anomaly, and orbital phase in the AK model. The secular evolution of $x(t)$ and $e_t(t)$ uses the 3PN equations. The waveform amplitude includes only the Newtonian quadrupolar $h_{20}$ and $h_{2,\pm2}$ modes.

![One-year AK Taiji response comparison](docs/figures/taiji_ak_tdi2_1yr_demo.png)



![One-year AK Taiji A-channel zoom](docs/figures/taiji_ak_tdi2_1yr_demo_A_zoom.png)

## Orbit Models and Data Sources

GWDelta can build FastLISAResponse-compatible orbit objects from the following `base` options:

| `base`            | Detector/orbit        | Source                                                       |
| ----------------- | --------------------- | ------------------------------------------------------------ |
| `taiji-accurate`  | Taiji realistic orbit | `MicroSateOrbit.hdf5` from [`TriangleDataCenter/Triangle-Simulator/OrbitData/MicroSateOrbitEclipticTCB`](https://github.com/TriangleDataCenter/Triangle-Simulator/tree/main/OrbitData/MicroSateOrbitEclipticTCB) |
| `taiji-triangle`  | Taiji equal-arm orbit | Samples from [`TriangleDataCenter/Triangle-Simulator/OrbitData/TaijiEqualArmOrbit`](https://github.com/TriangleDataCenter/Triangle-Simulator/tree/main/OrbitData/TaijiEqualArmOrbit) |
| `esa`             | LISA realistic orbit  | `ESAOrbits` from [`LISAanalysistools`](https://github.com/mikekatz04/LISAanalysistools) |
| `equal-armlength` | LISA equal-arm orbit  | `EqualArmlengthOrbits` from [`LISAanalysistools`](https://github.com/mikekatz04/LISAanalysistools) |
| `bbo-stage1-toy`  | BBO toy orbit         | Internal rigid heliocentric Stage-I toy model                |
| `tianqin-toy`     | TianQin toy orbit     | Internal rigid geocentric toy model                          |
| `file`            | User orbit            | Sampled NPZ/CSV orbit data                                   |

The BBO and TianQin entries are response-test toy orbits, not mission ephemerides.

Taiji orbit files are not bundled. Download the data from the links above and pass the directory through `orbit_dir`, or set `GWDELTA_TAIJI_ACCURATE_ORBIT_DIR`, `GWDELTA_TAIJI_TRIANGLE_ORBIT_DIR`, or `GWDELTA_ORBIT_DATA_DIR`.

**Warning:** The Taiji orbit files use the reverse `1,2,3` spacecraft ordering from the analytic response formulas in this code; GWDelta relabels spacecraft `1` and `2` and the corresponding light-time links internally when building the analytic-comparison orbit.

GWDelta can also generate simple equal-arm orbits directly from a reference triangle in the realistic Taiji orbit. First build the realistic Taiji orbit and relabel it to the standard TDI convention, then interpolate the three spacecraft positions at `reference_time_s`.

For short-duration analytic response checks, use the static helper. It builds one fixed equal-arm triangle with the same reference center, sets the effective arm length to the median reference arm length, and fits the analytic triangle orientation:

```python
from gwdelta import make_static_equal_arm_orbits_from_reference

simple_orbits, match = make_static_equal_arm_orbits_from_reference(
    reference_positions_m,
    duration_s=duration_s,
    reference_time_s=reference_time_s,
    center_at_reference=True,
    force_backend="cuda12x",
)
```

For year-long time-domain comparisons, use the dynamic helper. It matches the center, arm length, and analytic triangle orientation at `reference_time_s`, then lets the simple equal-arm Taiji-like orbit evolve with the same sidereal-year guiding-center phase:

```python
from gwdelta import make_dynamic_equal_arm_orbits_from_reference

simple_orbits, match = make_dynamic_equal_arm_orbits_from_reference(
    reference_positions_m,
    duration_s=duration_s,
    reference_time_s=reference_time_s,
    orbit_dt=600.0,
    force_backend="cuda12x",
)
```

The returned `match` records the reference positions, reference center, effective arm length, orientation parameters, guiding-center radius/phase, sampling cadence, and fit residual.

Orbit parameters can be changed through `make_orbits_from_spec`:

```python
from gwdelta import make_orbits_from_spec

orbits = make_orbits_from_spec(
    {
        "base": "taiji-accurate",
        "orbit_dir": "path/to/MicroSateOrbitEclipticTCB",
        "orbit_dt": 600.0,
        "time_offset": 0.0,
        "center_phase_deg": 20.0,
        "rotate_z_deg": 0.0,
        "translation_m": [0.0, 0.0, 0.0],
        "scale": 1.0,
    },
    duration=86400.0,
    force_backend="cpu",
)
```

Set `base` explicitly. The default values for the other optional orbit parameters are:

- `orbit_dt=600 s`;
- `time_offset=0`;
- `rotate_z_deg=0`;
- `translation_m=[0,0,0]`;
- `scale=1`;
- `armlength_m=None`, meaning use the source orbit value;
- `links=[12,23,31,13,32,21]`;
- `use_project_phase_defaults=True`.

Project phase defaults align LISA/equal-arm orbits to a center phase of `-20 deg` at local `t=0`, and Taiji orbits to `+20 deg`. Set `use_project_phase_defaults=False` to keep the raw orbit-file epoch.

Family-specific defaults:

- `taiji-accurate`: arm length defaults to the median file light time times `c`.
- `taiji-triangle`: nominal arm length is `3.0e9 m`.
- `bbo-stage1-toy`: `armlength_m=5.0e7`, guiding-center radius `1 AU`, center phase `-20 deg`, cartwheel period one sidereal year, cartwheel phase `90 deg`, detector-plane normal inclination `60 deg`; see [arXiv:gr-qc/0506015](https://arxiv.org/abs/gr-qc/0506015).
- `tianqin-toy`: geocentric radius `1.0e8 m`, arm length `sqrt(3) * 1.0e8 m`, guiding-center radius `1 AU`, fixed plane normal at longitude `120.5 deg` and latitude `-4.7 deg`; see [arXiv:2012.03260](https://arxiv.org/abs/2012.03260).

## TDI Options

The time-domain interface separates the TDI delay combination from the output channel basis:

- `tdi="1st generation"`: first-generation Michelson-style ordinary triplet; see [arXiv:gr-qc/0409034](https://arxiv.org/abs/gr-qc/0409034).
- `tdi="2nd generation"`: second-generation Michelson-style ordinary triplet; see [arXiv:gr-qc/0310017](https://arxiv.org/abs/gr-qc/0310017).
- `tdi="hybrid relay"`: hybrid Relay ordinary triplet; see [arXiv:2403.01490](https://arxiv.org/abs/2403.01490).
- `tdi=[...]`: a custom list of FastLISAResponse delay-term dictionaries.
- `tdi_chan="XYZ"`: return the three ordinary channels using the existing FastLISAResponse output names.
- `tdi_chan="AET"`: rotate the selected ordinary triplet to three optimal channels; see [arXiv:gr-qc/0209039](https://arxiv.org/abs/gr-qc/0209039).
- `tdi_chan="AE"`: return only the first two rotated channels.

Examples:

```python
from gwdelta import FastLISAResponseTDI

michelson = FastLISAResponseTDI(
    orbits=orbits,
    tdi="2nd generation",
    tdi_chan="AE",
)

hybrid_relay = FastLISAResponseTDI(
    orbits=orbits,
    tdi="hybrid relay",
    tdi_chan="AET",
)
```

The `tdi_chan` selector keeps the existing output naming convention. `XYZ` means the selected ordinary triplet before A/E/T rotation; the actual delay combination is selected by `tdi`.
