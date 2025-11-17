#!/usr/bin/env python3
"""Compose synthetic reconstruction + voltage comparison figures into PNG/SVG."""

from __future__ import annotations

import argparse
import base64
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def parse_length(value: str | None) -> float:
    if value is None:
        raise ValueError("Missing width/height in SVG")
    value = value.strip()
    units = {
        "px": 1.0,
        "pt": 96.0 / 72.0,
        "in": 96.0,
        "cm": 96.0 / 2.54,
        "mm": 96.0 / 25.4,
    }
    for suffix, scale in units.items():
        if value.endswith(suffix):
            return float(value[:-len(suffix)]) * scale
    return float(value)


def load_svg(path: Path) -> tuple[ET.Element, float, float]:
    tree = ET.parse(path)
    root = tree.getroot()
    width = parse_length(root.get("width") or root.get("viewBox", "0 0 1 1").split()[2])
    height = parse_length(root.get("height") or root.get("viewBox", "0 0 1 1").split()[3])
    return root, width, height


def compose_png(top_png: Path, bottom_png: Path, output_path: Path) -> None:
    top_img = Image.open(top_png).convert("RGB")
    bottom_img = Image.open(bottom_png).convert("RGB")
    canvas_width = max(top_img.width, bottom_img.width)
    canvas_height = top_img.height + bottom_img.height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    canvas.paste(top_img, ((canvas_width - top_img.width) // 2, 0))
    canvas.paste(bottom_img, ((canvas_width - bottom_img.width) // 2, top_img.height))
    canvas.save(output_path, dpi=(300, 300))


def compose_svg(top_png: Path, bottom_svg: Path, output_path: Path) -> None:
    root, svg_width, svg_height = load_svg(bottom_svg)
    top_img = Image.open(top_png).convert("RGB")
    combined_width = max(top_img.width, svg_width)
    scale = combined_width / svg_width
    combined_height = top_img.height + svg_height * scale

    svg_root = ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "xmlns:xlink": XLINK_NS,
            "width": f"{combined_width}",
            "height": f"{combined_height}",
            "viewBox": f"0 0 {combined_width} {combined_height}",
        },
    )

    with open(top_png, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    image_elem = ET.SubElement(
        svg_root,
        "image",
        {
            "x": "0",
            "y": "0",
            "width": f"{top_img.width}",
            "height": f"{top_img.height}",
            "preserveAspectRatio": "xMidYMid meet",
        },
    )
    image_elem.set(f"{{{XLINK_NS}}}href", f"data:image/png;base64,{encoded}")

    defs = []
    delta_group = ET.SubElement(
        svg_root,
        "g",
        {"transform": f"translate(0,{top_img.height}) scale({scale})"},
    )
    for child in root:
        tag = child.tag.split("}")[-1]
        if tag == "defs":
            defs.append(copy.deepcopy(child))
        else:
            delta_group.append(copy.deepcopy(child))
    for d in defs:
        svg_root.insert(0, d)

    tree = ET.ElementTree(svg_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=Path, required=True, help="Path to reconstruction comparison PNG")
    parser.add_argument("--bottom-svg", type=Path, required=True, help="Path to edited deltaV SVG")
    parser.add_argument("--bottom-png", type=Path, help="Path to PNG version of deltaV plot")
    parser.add_argument("--output-prefix", type=Path, default=Path("results/simulation_parity/run02/synthetic_combined"),
                        help="Prefix for combined outputs (without extension)")
    args = parser.parse_args()

    bottom_png = args.bottom_png or args.bottom_svg.with_suffix(".png")
    if not bottom_png.exists():
        raise FileNotFoundError(f"Bottom PNG not found: {bottom_png}. Please export your SVG to PNG first.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    compose_png(args.top, bottom_png, args.output_prefix.with_suffix(".png"))
    compose_svg(args.top, args.bottom_svg, args.output_prefix.with_suffix(".svg"))
    print(f"Combined figures written to {args.output_prefix.with_suffix('.png')} and .svg")


if __name__ == "__main__":
    main()
