#!/usr/bin/env python3
"""Create composite figure comparing ground truth, absolute reconstruction, and difference reconstruction."""
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path('results/simulation_parity/run02')
truth_path = ROOT / 'absolute' / 'phantom_ground_truth.png'
abs_path = ROOT / 'absolute' / 'reconstruction.png'
diff_path = ROOT / 'difference' / 'reconstruction.png'

images = [
    ("(a) Ground truth", truth_path),
    ("(b) PyEidors absolute reconstruction", abs_path),
    ("(c) PyEidors difference reconstruction", diff_path),
]

fig, axes = plt.subplots(1, len(images), figsize=(15, 4))
for ax, (title, path) in zip(axes, images):
    img = Image.open(path).convert('RGB')
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

fig.tight_layout()
(fig_path := ROOT / 'fig_absolute_vs_difference.png').parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=300)
fig.savefig(ROOT / 'fig_absolute_vs_difference.svg', format='svg')
plt.close(fig)
print(f"Saved figure to {fig_path} and corresponding SVG.")
