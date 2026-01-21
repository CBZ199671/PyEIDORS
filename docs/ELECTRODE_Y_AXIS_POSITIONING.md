# Electrode Y-Axis Positioning

This document describes the electrode positioning convention used in PyEIDORS, where the first electrode center is precisely located on the positive Y-axis (90°).

## Positioning Convention

### Default Behavior
- **First electrode center**: Precisely at Y-axis positive direction (90°)
- **Numbering order**: Counter-clockwise
- **Compatible**: Works with any number of electrodes (8, 16, 32, ...)

### Algorithm Implementation

```python
@property
def positions(self) -> List[Tuple[float, float]]:
    """Calculate start and end angles for each electrode."""
    electrode_size = 2 * pi / self.L * self.coverage
    gap_size = 2 * pi / self.L * (1 - self.coverage)

    # First electrode center at Y-axis positive (π/2)
    first_electrode_center = pi / 2 + self.rotation
    first_electrode_start = first_electrode_center - electrode_size / 2

    positions = []
    for i in range(self.L):
        total_space_per_electrode = electrode_size + gap_size
        start = first_electrode_start + i * total_space_per_electrode
        end = start + electrode_size
        positions.append((start, end))

    if not self.anticlockwise:
        positions[1:] = positions[1:][::-1]

    return positions
```

## Electrode Distribution Example

### 8-Electrode System (50% coverage)
| Electrode | Angle | Cartesian Coordinates |
|-----------|-------|----------------------|
| 1 | 90° | (0.000, 1.000) - Y+ axis |
| 2 | 135° | (-0.707, 0.707) |
| 3 | 180° | (-1.000, 0.000) - X- axis |
| 4 | 225° | (-0.707, -0.707) |
| 5 | 270° | (0.000, -1.000) - Y- axis |
| 6 | 315° | (0.707, -0.707) |
| 7 | 360° | (1.000, 0.000) - X+ axis |
| 8 | 405° | (0.707, 0.707) |

### 16-Electrode System
- Electrode spacing: 22.5° (360°/16)
- First electrode center: 90.000000°

## Usage

### Basic Usage
```python
from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

# Create 16-electrode configuration (first electrode at top)
electrodes = ElectrodePosition(L=16, coverage=0.5)
positions = electrodes.positions

# Verify first electrode center
first_center = (positions[0][0] + positions[0][1]) / 2
print(f"First electrode center: {first_center * 180 / np.pi:.6f}°")  # 90.000000°
```

### Custom Rotation
```python
# Rotate 45° from Y-axis, first electrode center at 135°
electrodes_rotated = ElectrodePosition(L=8, coverage=0.5, rotation=np.pi/4)
```

### Quick Mesh Generation
```python
from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh

# Create mesh with Y-axis starting electrode configuration
mesh = create_eit_mesh(n_elec=16, electrode_coverage=0.5)
```

## Precision Verification

The implementation achieves machine-precision accuracy:
- **Angular error**: < 1e-15 rad
- **Coordinate precision**: x < 6.12e-17, y = 1.0000000000

## Rationale

This convention provides several benefits:

1. **Intuitive**: First electrode at "top" (12 o'clock position)
2. **Medical standard**: Consistent with medical EIT device conventions
3. **Mathematical clarity**: Aligns with standard polar coordinate conventions
4. **Backward compatible**: `rotation` parameter allows adjustment if needed

## Migration from X-axis Convention

If your existing code assumes X-axis starting position, you can restore the old behavior:

```python
# Set rotation=-π/2 to start from X-axis
electrodes = ElectrodePosition(L=16, coverage=0.5, rotation=-np.pi/2)
```
