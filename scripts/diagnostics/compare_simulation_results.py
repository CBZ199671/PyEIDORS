import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

ROOT = Path('results/simulation_parity/run02')
meas = np.loadtxt(ROOT / 'synthetic_forward_data.csv', delimiter=',', skiprows=1, usecols=2)
py_pred = np.loadtxt(ROOT / 'difference' / 'predicted_difference.csv', delimiter=',')
eidors = np.loadtxt(ROOT / 'eidors_vs_pyeidors_difference.csv', delimiter=',')
eidors_pred = eidors[:,1]
idx = np.arange(1, meas.size + 1)

fig, ax = plt.subplots(figsize=(12, 4))

# ΔV traces
ax.plot(idx, meas, label='Measured ΔV', color='black', linewidth=1)
ax.plot(idx, py_pred, label='PyEIDORS ΔV_pred', color='tab:green', linestyle='--')
ax.plot(idx, eidors_pred, label='EIDORS ΔV_pred', color='tab:red')
ax.set_xlabel('Measurement index')
ax.set_ylabel('Voltage (V)')
ax.set_title('Boundary voltage differences')
ax.legend(loc='best')
ax.grid(False)

fig.tight_layout()
fig.savefig(ROOT / 'deltaV_comparison.png', dpi=300)
fig.savefig(ROOT / 'deltaV_comparison.svg', format='svg')
plt.close(fig)

# Reconstruction comparison
def prepare_image(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


image_specs = [
    ('Ground truth', ROOT / 'difference' / 'phantom_ground_truth.png'),
    ('PyEIDORS', ROOT / 'difference' / 'reconstruction.png'),
    ('EIDORS', ROOT / 'eidors_reconstruction.png'),
]

fig, axes = plt.subplots(1, len(image_specs), figsize=(15, 4))
for ax, (title, path) in zip(axes, image_specs):
    img = prepare_image(path)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
fig.tight_layout()
fig.savefig(ROOT / 'reconstruction_comparison.png', dpi=300)
fig.savefig(ROOT / 'reconstruction_comparison.svg', format='svg')
plt.close(fig)
