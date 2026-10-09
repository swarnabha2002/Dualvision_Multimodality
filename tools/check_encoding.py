"""
CPU-only sanity check on relative thermal encoding, using real M3FD images.

Run this before spending any GPU time. It answers three questions:

  1. Do relative-encoded images still look like recognisable scenes?
     If they look like noise, a frozen VLM will do badly on them and we
     need a gentler encoding before wasting cluster hours finding out.

  2. How much does each encoding actually move under a gamma change?
     The earlier numbers came from a synthetic image. These come from
     your data.

  3. Does the encoding clip? The formula is

         out = 128 + SCALE * (x - reference) / spread

     so anything further than (128 / SCALE) spreads from the reference
     saturates at 0 or 255 and its detail is destroyed. Hot engines are
     exactly the pixels most at risk. If the clipped fraction is high,
     SCALE is too large.

Outputs:
  data/M3FD/encoding_check/grid_<stem>.png   visual comparisons
  data/M3FD/encoding_check/scale_sweep.png   effect of SCALE
  plus a statistics table printed to the terminal

Needs numpy and Pillow only. No torch, no GPU.
"""

import os
import sys
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

M3FD = os.path.join("data", "M3FD")
IR = os.path.join(M3FD, "ir")
VI = os.path.join(M3FD, "vi")
OUTDIR = os.path.join(M3FD, "encoding_check")

STANDARD_SIZE = (1024, 768)
PERTURBATIONS = ["none", "offset+20", "gain1.2", "gamma0.7", "gamma1.4"]


# ------------------------------------------------------------------ imaging
# Kept identical to run_pilot.py. If you change one, change both.

def read_thermal(stem):
    arr = np.array(Image.open(os.path.join(IR, stem + ".png")))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64)


def perturb(therm, kind):
    if kind == "none":
        return therm
    if kind.startswith("offset"):
        return therm + float(kind.replace("offset", ""))
    if kind.startswith("gamma"):
        g = float(kind.replace("gamma", ""))
        x = np.clip(therm, 0, 255) / 255.0
        return np.power(x, g) * 255.0
    if kind.startswith("gain"):
        g = float(kind.replace("gain", ""))
        return (therm - therm.mean()) * g + therm.mean()
    raise ValueError("unknown perturbation: " + kind)


def encode_abs(therm):
    return np.clip(therm, 0, 255)


def encode_rel(therm, ref_pct=50.0, scale=40.0):
    reference = np.percentile(therm, ref_pct)
    mad = np.median(np.abs(therm - reference))
    spread = max(mad, 1.0)
    return np.clip(128.0 + scale * (therm - reference) / spread, 0, 255)


def encode_rel_raw(therm, ref_pct=50.0, scale=40.0):
    """Same but WITHOUT clipping, so we can measure how much would clip."""
    reference = np.percentile(therm, ref_pct)
    mad = np.median(np.abs(therm - reference))
    spread = max(mad, 1.0)
    return 128.0 + scale * (therm - reference) / spread


def to_pil(arr, size=None):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(np.stack([a, a, a], axis=-1))
    if size:
        im = im.resize(size)
    return im


# ------------------------------------------------------------------ sheets
def grid_sheet(stem, out_path, ref_pct, scale, cell=248):
    """
    One row per encoding, one column per perturbation, plus the RGB image
    for reference. Look at this and judge whether the scene survives.
    """
    therm = read_thermal(stem)
    cols = 1 + len(PERTURBATIONS)
    rows = 2
    cap_h = 22
    W = cols * cell
    H = rows * (cell + cap_h) + cap_h

    sheet = Image.new("RGB", (W, H), (16, 16, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "stem " + stem + "   ref_pct=" + str(ref_pct)
              + "   scale=" + str(scale), fill=(210, 215, 230))

    ar = (cell, int(cell * STANDARD_SIZE[1] / STANDARD_SIZE[0]))

    # column 0: the RGB image, for both rows
    try:
        rgb = Image.open(os.path.join(VI, stem + ".png")).convert("RGB").resize(ar)
        for r in range(rows):
            y = cap_h + r * (cell + cap_h)
            sheet.paste(rgb, (0, y))
            draw.text((4, y + ar[1] + 3), "RGB (reference)", fill=(150, 158, 178))
    except OSError:
        pass

    for c, pert in enumerate(PERTURBATIONS, start=1):
        p = perturb(therm, pert)
        for r, enc in enumerate(["abs", "rel"]):
            arr = encode_abs(p) if enc == "abs" else encode_rel(p, ref_pct, scale)
            y = cap_h + r * (cell + cap_h)
            sheet.paste(to_pil(arr, ar), (c * cell, y))
            draw.text((c * cell + 4, y + ar[1] + 3),
                      enc + "  " + pert, fill=(210, 160, 70) if enc == "abs" else (110, 180, 220))

    sheet.save(out_path)
    return out_path


def scale_sweep_sheet(stem, out_path, ref_pct, scales, cell=248):
    """How SCALE trades contrast against clipping."""
    therm = read_thermal(stem)
    ar = (cell, int(cell * STANDARD_SIZE[1] / STANDARD_SIZE[0]))
    cap_h = 22
    W = len(scales) * cell
    H = cell + 2 * cap_h

    sheet = Image.new("RGB", (W, H), (16, 16, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "SCALE sweep, stem " + stem, fill=(210, 215, 230))

    for c, s in enumerate(scales):
        arr = encode_rel(therm, ref_pct, s)
        unclipped = encode_rel_raw(therm, ref_pct, s)
        clipped = float(((unclipped < 0) | (unclipped > 255)).mean())
        sheet.paste(to_pil(arr, ar), (c * cell, cap_h))
        draw.text((c * cell + 4, cap_h + ar[1] + 3),
                  "scale " + str(s) + "   clipped " + str(round(100 * clipped, 1)) + "%",
                  fill=(110, 180, 220))
    sheet.save(out_path)
    return out_path


# ------------------------------------------------------------------ stats
def statistics(stems, ref_pct, scale):
    """Aggregate over real images: movement, structure, clipping."""
    move = defaultdict(list)
    corr = defaultdict(list)
    clip_frac = []
    contrast = defaultdict(list)

    for stem in stems:
        therm = read_thermal(stem)
        clean_a = encode_abs(therm)
        clean_r = encode_rel(therm, ref_pct, scale)

        unclipped = encode_rel_raw(therm, ref_pct, scale)
        clip_frac.append(float(((unclipped < 0) | (unclipped > 255)).mean()))

        contrast["abs"].append(float(clean_a.std()))
        contrast["rel"].append(float(clean_r.std()))

        for pert in PERTURBATIONS:
            if pert == "none":
                continue
            p = perturb(therm, pert)
            a = encode_abs(p)
            r = encode_rel(p, ref_pct, scale)
            move[("abs", pert)].append(float(np.abs(a - clean_a).mean()))
            move[("rel", pert)].append(float(np.abs(r - clean_r).mean()))
            corr[("abs", pert)].append(float(np.corrcoef(a.ravel(), clean_a.ravel())[0, 1]))
            corr[("rel", pert)].append(float(np.corrcoef(r.ravel(), clean_r.ravel())[0, 1]))

    return move, corr, clip_frac, contrast


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=6, help="how many visual grids to save")
    ap.add_argument("--stats-images", type=int, default=150, help="images used for the numbers")
    ap.add_argument("--ref-pct", type=float, default=50.0)
    ap.add_argument("--scale", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Project root: " + os.getcwd())

    if not os.path.isdir(IR):
        print("Missing " + IR + " -- run get_m3fd.py first.")
        sys.exit(1)

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(IR, "*.png")))
    eligible = [s for s in stems
                if Image.open(os.path.join(IR, s + ".png")).size == STANDARD_SIZE]
    print("usable thermal images: " + str(len(eligible)))
    rng.shuffle(eligible)

    os.makedirs(OUTDIR, exist_ok=True)

    # ---- visual sheets
    print("")
    print("=" * 62)
    print("  VISUAL SHEETS")
    print("=" * 62)
    for stem in eligible[:args.sheets]:
        path = os.path.join(OUTDIR, "grid_" + stem + ".png")
        grid_sheet(stem, path, args.ref_pct, args.scale)
        print("  " + path)

    sweep_stem = eligible[0]
    sweep_path = os.path.join(OUTDIR, "scale_sweep.png")
    scale_sweep_sheet(sweep_stem, sweep_path, args.ref_pct, [20, 30, 40, 60, 90])
    print("  " + sweep_path)

    # ---- statistics
    sample = eligible[:args.stats_images]
    print("")
    print("=" * 62)
    print("  NUMBERS OVER " + str(len(sample)) + " REAL IMAGES")
    print("=" * 62)
    move, corr, clip_frac, contrast = statistics(sample, args.ref_pct, args.scale)

    print("")
    print("  How far the image moves under each perturbation")
    print("  (mean absolute pixel change from the unperturbed version)")
    print("")
    print("  perturbation".ljust(18) + "abs".rjust(10) + "rel".rjust(10) + "   verdict")
    print("  " + "-" * 52)
    for pert in PERTURBATIONS:
        if pert == "none":
            continue
        a = float(np.mean(move[("abs", pert)]))
        r = float(np.mean(move[("rel", pert)]))
        if r < 0.01:
            verdict = "exact (algebra, not evidence)"
        elif r < a * 0.3:
            verdict = "much more stable"
        elif r < a * 0.8:
            verdict = "somewhat more stable"
        else:
            verdict = "no real advantage"
        print("  " + pert.ljust(16) + (str(round(a, 2))).rjust(10)
              + (str(round(r, 2))).rjust(10) + "   " + verdict)

    print("")
    print("  Structure retained (correlation with the unperturbed image)")
    print("")
    print("  perturbation".ljust(18) + "abs".rjust(10) + "rel".rjust(10))
    print("  " + "-" * 40)
    for pert in PERTURBATIONS:
        if pert == "none":
            continue
        a = float(np.mean(corr[("abs", pert)]))
        r = float(np.mean(corr[("rel", pert)]))
        print("  " + pert.ljust(16) + (str(round(a, 4))).rjust(10)
              + (str(round(r, 4))).rjust(10))

    cf = np.array(clip_frac)
    print("")
    print("  Clipping in the relative encoding at scale=" + str(args.scale))
    print("    mean " + str(round(100 * float(cf.mean()), 2)) + "%"
          + "   median " + str(round(100 * float(np.median(cf)), 2)) + "%"
          + "   worst " + str(round(100 * float(cf.max()), 2)) + "%")
    if cf.mean() > 0.05:
        print("    >> More than 5% of pixels saturate on average. Hot objects are")
        print("       exactly what saturates, so detail you need is being lost.")
        print("       Look at scale_sweep.png and try a smaller --scale.")
    else:
        print("    >> Acceptable. Little detail is being destroyed.")

    print("")
    print("  Contrast (std dev of the encoded image)")
    print("    abs " + str(round(float(np.mean(contrast["abs"])), 1))
          + "    rel " + str(round(float(np.mean(contrast["rel"])), 1)))
    print("    Similar numbers mean rel images are about as visually rich as abs.")
    print("    Much lower for rel means the encoding is flattening the scene.")

    print("")
    print("=" * 62)
    print("  NOW GO AND LOOK AT THE SHEETS")
    print("=" * 62)
    print("  Open " + OUTDIR)
    print("")
    print("  Top row is absolute encoding, bottom row is relative, and the")
    print("  leftmost column is the RGB image so you know what the scene is.")
    print("")
    print("  The question: in the rel row, can you still recognise people,")
    print("  vehicles and road? If yes, the pilot is worth GPU time. If the")
    print("  rel images look washed out or noisy, tell me what you see and we")
    print("  will adjust scale or the reference percentile before you run it.")


if __name__ == "__main__":
    main()
