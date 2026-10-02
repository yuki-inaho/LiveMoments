"""Prepare reproducible synthetic triplets without using any user photographs."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_utils.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    y, x = np.mgrid[0:768, 0:768]
    pixels = np.stack(
        [(x // 3) % 256, (y // 3) % 256, ((x // 48 + y // 48) % 2) * 180 + 30], axis=-1
    ).astype("uint8")
    high = Image.fromarray(pixels)
    low = high.filter(ImageFilter.GaussianBlur(2))
    for part, image in [("lr", low), ("ref", high), ("ref_lr", low)]:
        (args.out / part).mkdir()
        image.save(args.out / part / "sample.png")
    cfg = load_config(ROOT / "config/inference_example.yml")
    for part in ["lr", "ref", "ref_lr"]:
        cfg[part + "_path"] = str((args.out / part).resolve())
    (args.out / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Synthetic triplets and config created; no GPU inference performed")


if __name__ == "__main__":
    main()
