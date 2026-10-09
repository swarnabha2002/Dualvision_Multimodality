"""
Which spread estimator survives gamma best?

The relative encoding is

    out = 128 + scale * (x - reference) / spread

`spread` is one number saying how varied the image is. It becomes the
divisor, so if it moves when the sensor changes, the whole encoding moves
with it. That is what breaks on gamma1.4 in scenes with a large sunlit
surface: the divisor swings, and the extremes blow out to white.

Three ways to measure spread, same everything else:

  mad   median absolute deviation. Median distance of a pixel from the
        reference. One point on a distribution -- move it and the divisor
        moves with it.

  ipr   inter-percentile range. The gap between the 85th and 15th
        percentiles. Two points far apart, so when gamma compresses one
        part of the histogram and stretches another, the changes partly
        cancel.

  std   standard deviation. The textbook answer. Squaring distances makes
        extreme pixels count heavily, so a hot building can dominate --
        but a reviewer will ask, so measure it rather than argue.

Because the three produce numbers on different scales, each one gets its
own `scale` value, calibrated so all three give roughly equal contrast on
clean images. Otherwise you would be comparing contrast, not stability.

Outputs
  a table of movement under each perturbation, per estimator
  a table of clipping, per estimator
  data/M3FD/spread_check/spread_<stem>.png  visual comparison

CPU only. numpy and Pillow.
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
OUTDIR = os.path.join(M3FD, "spread_check")

STANDARD_SIZE = (1024, 768)
PERTURBATIONS = ["none", "offset+20", "gain1.2", "gamma0.7", "gamma1.4"]
ESTIMATORS = ["mad", "ipr", "std"]


# ------------------------------------------------------------------ imaging
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


def get_spread(therm, reference, method, lo_pct=15.0, hi_pct=85.0):
    """The one line that differs between the three variants."""
    if method == "mad":
        return max(float(np.median(np.abs(therm - reference))), 1.0)
    if method == "ipr":
        hi = float(np.percentile(therm, hi_pct))
        lo = float(np.percentile(therm, lo_pct))
        return max(hi - lo, 1.0)
    if method == "std":
        return max(float(therm.std()), 1.0)
    raise ValueError("unknown spread method: " + method)


def encode_rel(therm, method, scale, ref_pct=50.0, clip=True):
    reference = np.percentile(therm, ref_pct)
    spread = get_spread(therm, reference, method)
    out = 128.0 + scale * (therm - reference) / spread
    return np.clip(out, 0, 255) if clip else out


def encode_abs(therm):
    return np.clip(therm, 0, 255)


def to_pil(arr, size=None):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(np.stack([a, a, a], axis=-1))
    return im.resize(size) if size else im


# ------------------------------------------------------------------ calibration
def calibrate_scales(stems, target_contrast, ref_pct):
    """
    Put the three estimators on equal footing.

    MAD, IPR and standard deviation return numbers of different magnitude,
    so the same `scale` would give each a different contrast. Comparing
    them then would measure contrast rather than stability. Here each
    estimator gets the scale that makes its clean images average the same
    contrast as the others.
    """
    print("  calibrating scales so all three start at contrast ~"
          + str(target_contrast) + " ...")
    scales = {}
    for method in ESTIMATORS:
        # Contrast is linear in scale, so measure at 1.0 and solve directly.
        contrasts = []
        for stem in stems:
            therm = read_thermal(stem)
            contrasts.append(float(encode_rel(therm, method, 1.0, ref_pct,
                                              clip=False).std()))
        unit = float(np.mean(contrasts))
        scales[method] = round(target_contrast / max(unit, 1e-6), 2)
        print("    " + method.ljust(5) + " scale " + str(scales[method]))
    return scales


# ------------------------------------------------------------------ sheets
def sheet(stem, out_path, scales, ref_pct, cell=248):
    """One row per estimator plus one for absolute, one column per perturbation."""
    therm = read_thermal(stem)
    rows = ["abs"] + ESTIMATORS
    cols = 1 + len(PERTURBATIONS)
    cap_h = 22
    ar = (cell, int(cell * STANDARD_SIZE[1] / STANDARD_SIZE[0]))
    W = cols * cell
    H = len(rows) * (ar[1] + cap_h) + cap_h

    s = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(s)
    d.text((6, 5), "stem " + stem + "   ref_pct=" + str(ref_pct)
           + "   scales " + str(scales), fill=(210, 215, 230))

    try:
        rgb = Image.open(os.path.join(VI, stem + ".png")).convert("RGB").resize(ar)
    except OSError:
        rgb = None

    for r, method in enumerate(rows):
        y = cap_h + r * (ar[1] + cap_h)
        if rgb is not None:
            s.paste(rgb, (0, y))
            d.text((4, y + ar[1] + 3), "RGB", fill=(150, 158, 178))
        for c, pert in enumerate(PERTURBATIONS, start=1):
            p = perturb(therm, pert)
            if method == "abs":
                arr = encode_abs(p)
                colour = (210, 160, 70)
            else:
                arr = encode_rel(p, method, scales[method], ref_pct)
                colour = (110, 180, 220)
            s.paste(to_pil(arr, ar), (c * cell, y))
            d.text((c * cell + 4, y + ar[1] + 3), method + "  " + pert, fill=colour)

    s.save(out_path)
    return out_path


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-images", type=int, default=150)
    ap.add_argument("--sheets", type=int, default=4)
    ap.add_argument("--ref-pct", type=float, default=50.0)
    ap.add_argument("--target-contrast", type=float, default=45.0,
                    help="contrast all three estimators are calibrated to")
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
    sample = eligible[:args.stats_images]

    os.makedirs(OUTDIR, exist_ok=True)

    print("")
    print("=" * 66)
    print("  CALIBRATION")
    print("=" * 66)
    scales = calibrate_scales(sample[:40], args.target_contrast, args.ref_pct)

    # ---- measure
    print("")
    print("  measuring over " + str(len(sample)) + " images ...")
    move = defaultdict(list)
    clip = defaultdict(list)
    contrast = defaultdict(list)
    abs_move = defaultdict(list)

    for n, stem in enumerate(sample):
        therm = read_thermal(stem)
        clean_abs = encode_abs(therm)
        clean = {m: encode_rel(therm, m, scales[m], args.ref_pct)
                 for m in ESTIMATORS}

        for m in ESTIMATORS:
            contrast[m].append(float(clean[m].std()))
            unclipped = encode_rel(therm, m, scales[m], args.ref_pct, clip=False)
            clip[m].append(float(((unclipped < 0) | (unclipped > 255)).mean()))

        for pert in PERTURBATIONS:
            if pert == "none":
                continue
            p = perturb(therm, pert)
            abs_move[pert].append(float(np.abs(encode_abs(p) - clean_abs).mean()))
            for m in ESTIMATORS:
                r = encode_rel(p, m, scales[m], args.ref_pct)
                move[(m, pert)].append(float(np.abs(r - clean[m]).mean()))

        if (n + 1) % 50 == 0:
            print("    " + str(n + 1) + " / " + str(len(sample)))

    # ---- report
    print("")
    print("=" * 66)
    print("  MOVEMENT UNDER PERTURBATION  (lower is better)")
    print("=" * 66)
    print("  mean absolute pixel change from the unperturbed version")
    print("")
    header = "  perturbation".ljust(18) + "abs".rjust(9)
    for m in ESTIMATORS:
        header += m.rjust(9)
    print(header)
    print("  " + "-" * 48)
    for pert in PERTURBATIONS:
        if pert == "none":
            continue
        line = ("  " + pert).ljust(18)
        line += str(round(float(np.mean(abs_move[pert])), 2)).rjust(9)
        for m in ESTIMATORS:
            line += str(round(float(np.mean(move[(m, pert)])), 2)).rjust(9)
        print(line)

    gammas = [p for p in PERTURBATIONS if p.startswith("gamma")]
    print("")
    print("  gamma only (the perturbation that does not cancel by algebra):")
    best, best_v = None, None
    for m in ESTIMATORS:
        v = float(np.mean([np.mean(move[(m, p)]) for p in gammas]))
        print("    " + m.ljust(5) + str(round(v, 2)))
        if best_v is None or v < best_v:
            best, best_v = m, v
    abs_g = float(np.mean([np.mean(abs_move[p]) for p in gammas]))
    print("    abs  " + str(round(abs_g, 2)) + "   (the baseline being beaten)")

    print("")
    print("=" * 66)
    print("  CLIPPING AND CONTRAST AT MATCHED CONTRAST")
    print("=" * 66)
    print("  method".ljust(12) + "scale".rjust(8) + "clip mean".rjust(12)
          + "clip worst".rjust(12) + "contrast".rjust(11))
    print("  " + "-" * 53)
    for m in ESTIMATORS:
        c = np.array(clip[m])
        print("  " + m.ljust(10) + str(scales[m]).rjust(8)
              + (str(round(100 * float(c.mean()), 2)) + "%").rjust(12)
              + (str(round(100 * float(c.max()), 2)) + "%").rjust(12)
              + str(round(float(np.mean(contrast[m])), 1)).rjust(11))

    # ---- sheets
    print("")
    print("=" * 66)
    print("  VISUAL SHEETS")
    print("=" * 66)
    for stem in sample[:args.sheets]:
        print("  " + sheet(stem, os.path.join(OUTDIR, "spread_" + stem + ".png"),
                           scales, args.ref_pct))

    print("")
    print("=" * 66)
    print("  VERDICT")
    print("=" * 66)
    print("  Lowest gamma movement: " + best + "  (" + str(round(best_v, 2))
          + " against abs " + str(round(abs_g, 2)) + ")")
    print("")
    print("  Do not adopt it on this number alone. Check the sheets too --")
    print("  in particular whether the gamma1.4 panel still blows out on a")
    print("  scene with a large sunlit surface. If two estimators are close,")
    print("  prefer the one with less clipping.")
    print("")
    print("  Once you pick one, set it in run_pilot.py so the pilot and this")
    print("  check use the same encoding.")


if __name__ == "__main__":
    main()
