#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

from tools.coaster_gen import CoasterGenerator


def find_latest_download_image(downloads_dir: Path) -> Optional[Path]:
    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        candidates.extend(downloads_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Run PNG->SVG vectorization only")
    parser.add_argument("--input", help="Path to input image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Path to output SVG")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        downloads_dir = Path.home() / "Downloads"
        input_path = find_latest_download_image(downloads_dir)
        if input_path is None:
            raise SystemExit(f"No PNG/JPG files found in {downloads_dir}")

    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input image not found: {input_path}")

    image_bytes = input_path.read_bytes()
    generator = CoasterGenerator()
    svg_content = generator.vectorize_image(image_bytes, str(output_path.parent))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")

    print(f"Wrote SVG to: {output_path}")


if __name__ == "__main__":
    main()
