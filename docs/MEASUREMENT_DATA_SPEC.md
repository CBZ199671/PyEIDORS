# PyEIDORS Measurement Data Specification

This specification defines the format for real measurement data accepted by PyEIDORS inverse problem modules, ensuring consistency with the simulation interface. Hardware acquisition systems, host software, or data preprocessing pipelines only need to convert data to this format to interface directly with `EITSystem` and its dependent modules.

## 1. Dataset Components

A complete measurement dataset consists of two parts: **measurement matrix** and **metadata**.

| Name | Description |
| --- | --- |
| `measurements` | A float array of shape `(n_frames, n_meas_total)`, where `n_frames` represents the number of complete excitation cycles (e.g., one round of measurements after 16 excitation patterns are applied sequentially). For single-frame data, shape can be `(n_meas_total,)`. Unit: Volts (V). |
| `metadata` | Must contain configuration information related to measurements. See field definitions below. Recommended formats: YAML or JSON. |

Supported storage formats:

- **Recommended**: A single `.npz` file containing the `measurements` array and `metadata` dictionary serialized as a JSON string;
- Or: `measurements.csv` + `metadata.yaml` in the same directory.

Regardless of format, `metadata` must be complete and consistent with the measurement matrix.

## 2. Metadata Fields

To maintain consistency with `PatternConfig` and `StimMeasPatternManager` internal logic, `metadata` must contain at least the following fields:

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| `n_elec` | `int` | `16` | Number of electrodes. |
| `stim_pattern` | `str \| List[int]` | `"{ad}"` | Excitation pattern. Supports PyEIDORS shorthand notation or explicit electrode index lists. |
| `meas_pattern` | `str \| List[int]` | `"{ad}"` | Measurement pattern. |
| `drive_mode` | `str` | `"line_current_density"` | Drive semantics. Supported: `"line_current_density"`, `"total_current"`, `"normalized"`. |
| `drive_value` | `float` | `0.02` | Excitation magnitude associated with `drive_mode` (`A/m`, `A`, or normalized). |
| `geometry_scale_to_m` | `float` | `1.0` | Geometry scale to meters (`m=1.0`, `cm=0.01`, `mm=0.001`). |
| `electrode_length_m_override` | `float \| List[float]` | `null` | Optional physical electrode length(s) in meters. |
| `use_meas_current` | `bool` | `false` | Corresponds to `PatternConfig.use_meas_current`. |
| `use_meas_current_next` | `int` | `0` | Corresponds to `PatternConfig.use_meas_current_next`. |
| `rotate_meas` | `bool` | `true` | Whether measurements rotate with excitation. |
| `n_frames` | `int` | `1` | Number of complete excitation cycles, matching the first dimension of `measurements`. |
| `timestamp` | `str` | `"2025-06-29T20:00:52"` | (Optional) Acquisition time in ISO8601 format. |
| `sampling_rate_hz` | `float` | `1000.0` | (Optional) Sampling frequency for logging and tracking. |
| `notes` | `str` | `"Experiment #1, 20uA"` | (Optional) Additional notes. |

> Note: These fields correspond to all required information in `PatternConfig`, allowing direct reconstruction of the same excitation/measurement patterns upon loading. Additional hardware-related parameters may be added to `metadata`, but the meanings of the above fields should not be modified. The obsolete field `amplitude` is unsupported.

## 3. Measurement Matrix Ordering

The column order of the `measurements` array must follow the output order of `StimMeasPatternManager.apply_meas_pattern`:

1. Iterate through excitation patterns in generation order (internal electrode rotation + ring index);
2. For each excitation, expand the filtered measurement vector row by row according to current measurement pattern;
3. Result is a flattened vector of length `n_meas_total`.

If hardware outputs measurements in a 2D "excitation x channel" arrangement, reorder according to the above rules during conversion, or use `StimMeasPatternManager` helper methods for mapping (see interface example below).

## 4. Building PyEIDORS Data Objects

Once measurement matrix and metadata conform to this specification, use the `MeasurementDataset` helper class to quickly generate `EITData` instances:

```python
from pathlib import Path
import numpy as np
import yaml

from pyeidors.data.measurement_dataset import MeasurementDataset

# Load data
data_dir = Path("data/measurements")
measurements = np.loadtxt(data_dir / "2025-06-29-20-00-52.csv", delimiter=",")
metadata = yaml.safe_load((data_dir / "2025-06-29-20-00-52.yaml").read_text())

# Build dataset
dataset = MeasurementDataset.from_metadata(
    measurements=measurements,
    metadata=metadata,
    data_type="real"
)

# Generate EITData for inverse problem (default: read-only shared view)
eit_data = dataset.to_eit_data(frame_index=0)
# If a writable buffer is required:
eit_data_copy = dataset.to_eit_data(frame_index=0, copy_policy="copy")
```

`MeasurementDataset` automatically creates `PatternConfig` and `StimMeasPatternManager` based on metadata, validating measurement array shape and channel count. Mismatches raise exceptions with contextual information for easier debugging during data preprocessing.

`MeasurementDataset.to_eit_data` defaults to `copy_policy="view"` to reduce memory copies in reconstruction pipelines. Returned arrays are read-only views. If downstream code requires in-place edits, explicitly request `copy_policy="copy"` or call `.copy()` on the relevant array.

## 5. Managing Multi-Frame Data

When `n_frames > 1`, indicating multiple complete excitation cycles were acquired, `measurements` has shape `(n_frames, n_meas_total)`. You can:

- Use `MeasurementDataset.iter_frames()` to generate `EITData` frame by frame;
- Or explicitly specify `frame_index` in `to_eit_data`.
- `iter_frames(copy_policy="view")` is default and keeps shared read-only memory.

This is particularly important for time-series reconstruction (e.g., difference or time-difference analysis).

## 6. File Naming Convention

For traceability, the recommended naming format is:

```
YYYY-MM-DD-HH-MM-SS_<run_id>_<drive_mode>_<drive_value>_<frequency>
```

Example: `2025-06-29-20-00-52_1_lineCurrentDensity_5e-5Apm_1000Hz`

- Matching `.csv` and `.yaml/.json` files should appear in pairs;
- For `.npz` format, use this naming scheme as the filename prefix.

## 7. Responsibility Division

- **Hardware/Data Acquisition**: Convert raw data to the format required by this specification;
- **PyEIDORS Package**: Provide standard data structures (`EITData`), validation, and construction tools without coupling to hardware formats;
- **Downstream Algorithms/Applications**: Can assume all real measurement data conforms to this specification, focusing on reconstruction algorithms.

This specification allows the project to maintain a single stable interface, significantly reducing maintenance costs from hardware diversity.

## 8. End-to-End Example Script

After preparing standardized data, run `scripts/run_reconstruction_unified.py` for a difference reconstruction workflow:

```bash
python scripts/run_reconstruction_unified.py \
  --method gn-difference \
  --input-mode paired \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --metadata data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.yaml \
  --reference-col 0 \
  --target-col 2 \
  --lambda 1.0 \
  --output-root results/demo_output
```

For 3D cylindrical reconstruction (current release: `gn-difference` only), add:

```bash
python scripts/run_reconstruction_unified.py \
  --method gn-difference \
  --input-mode paired \
  --csv data/measurements/tank/sample.csv \
  --output-root results/demo_output_3d \
  --mesh-dim 3 \
  --mesh-height 0.2 \
  --radius 0.25 \
  --refinement 3
```

This script will:

- Validate measurement matrix dimensions and build `MeasurementDataset`;
- Set up `EITSystem` (preferring cached meshes, optionally using simplified mesh generator);
- In 3D mode, force `drive_mode=total_current` (2D-specific `line_current_density` is not used);
- Perform difference reconstruction using frame 0 as reference;
- Save measurement curves and conductivity reconstruction images to the output directory.
