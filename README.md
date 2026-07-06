# GWDelta

<img src="docs/figures/logo.png" alt="Taiji TDI response comparison" style="zoom: 25%;" />

GWDelta is a toolkit for fast response calculations for space-based gravitational-wave detectors, focusing on LISA-like triangular constellations.

The response code can run on CPU or on a CUDA-enabled `fastlisaresponse` / `lisatools` backend through `force_backend="cuda12x"`. Waveform-side helper paths can also use GPU acceleration where CuPy or Numba CUDA is available.

## Dependencies

The time-domain response path uses installed `fastlisaresponse` and `lisatools` backends. `fastlisaresponse` provides the `pyResponseTDI` engine; `lisatools` provides the orbit interface expected by that engine and the built-in LISA `esa` / `equal-armlength` orbit classes.

## Example

The example below compares a precessing quasi-circular SMBHB waveform generated with `SEOBNRv5PHM` and null displacement memory (including all $l=2$ modes computed perturbatively) against three Taiji response calculations:

- second-generation $A,E$ channels with a realistic Taiji orbit;
- second-generation $A,E$ channels with a static equal-arm orbit;
- an analytic static equal-arm frequency-domain response.

![Taiji TDI response comparison](docs/figures/taiji_static_tdi2_memory_demo.png)



<img src="docs/figures/taiji_ae_time_frequency.png" alt="Taiji A/E time-frequency map" style="zoom: 25%;" />

## Orbit Models and Data Sources

GWDelta can build FastLISAResponse-compatible orbit objects from the following `base` options:

| `base` | Detector/orbit | Source |
| --- | --- | --- |
| `taiji-accurate` | Taiji realistic orbit | `MicroSateOrbit.hdf5` from [`TriangleDataCenter/Triangle-Simulator/OrbitData/MicroSateOrbitEclipticTCB`](https://github.com/TriangleDataCenter/Triangle-Simulator/tree/main/OrbitData/MicroSateOrbitEclipticTCB) |
| `taiji-triangle` | Taiji equal-arm orbit | Samples from [`TriangleDataCenter/Triangle-Simulator/OrbitData/TaijiEqualArmOrbit`](https://github.com/TriangleDataCenter/Triangle-Simulator/tree/main/OrbitData/TaijiEqualArmOrbit) |
| `esa` | LISA realistic orbit | `ESAOrbits` from [`LISAanalysistools`](https://github.com/mikekatz04/LISAanalysistools) |
| `equal-armlength` | LISA equal-arm orbit | `EqualArmlengthOrbits` from [`LISAanalysistools`](https://github.com/mikekatz04/LISAanalysistools) |
| `bbo-stage1-toy` | BBO toy orbit | Internal rigid heliocentric Stage-I toy model |
| `tianqin-toy` | TianQin toy orbit | Internal rigid geocentric toy model |
| `file` | User orbit | Sampled NPZ/CSV orbit data |

The BBO and TianQin entries are response-test toy orbits, not mission ephemerides.

Taiji orbit files are not bundled. Download the data from the links above and pass the directory through `orbit_dir`, or set `GWDELTA_TAIJI_ACCURATE_ORBIT_DIR`, `GWDELTA_TAIJI_TRIANGLE_ORBIT_DIR`, or `GWDELTA_ORBIT_DATA_DIR`.

**Warning:** The Triangle-Simulator Taiji orbit files use the reverse `1,2,3` spacecraft ordering from the analytic response formulas in this code; GWDelta relabels spacecraft `1` and `2` and the corresponding light-time links internally when building the analytic-comparison orbit.

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

- `taiji-accurate`: samples `MicroSateOrbit.hdf5`; arm length defaults to the median file light time times `c`; see [arXiv:1807.09495](https://arxiv.org/abs/1807.09495).
- `taiji-triangle`: samples Triangle-Simulator equal-arm files; nominal arm length is `3.0e9 m`; see [arXiv:1707.09127](https://arxiv.org/abs/1707.09127).
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
