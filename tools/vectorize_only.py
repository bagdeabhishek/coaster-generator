#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "main.py"
if not MAIN_PATH.exists():
    raise SystemExit(f"main.py not found at {MAIN_PATH}")

spec = importlib.util.spec_from_file_location("coaster_main", MAIN_PATH)
if spec is None or spec.loader is None:
    raise SystemExit("Failed to load main.py")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

vectorize_image = module.vectorize_image


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
    svg_content = vectorize_image(image_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")

    print(f"Wrote SVG to: {output_path}")


if __name__ == "__main__":
    main()
