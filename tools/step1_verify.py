"""Step 1: verify what actually survived the 8-bit conversion in an RGB-thermal dataset.

CPU-only. numpy + Pillow only. Read-only with respect to the dataset.

Usage:
    python tools/step1_verify.py --name LLVIP \
        --rgb data/LLVIP_flat/LLVIP/visible \
        --thermal data/LLVIP_flat/LLVIP/infrared \
        --sample 300
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm")


def die(msg):
    print("")
    print("FATAL: " + msg)
    sys.exit(1)


def list_images(d, label):
    if not os.path.isdir(d):
        die("directory for " + label + " does not exist: " + d)
    names = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if not os.path.isfile(p):
            continue
        if n.lower().endswith(IMAGE_EXTS):
            names.append(n)
    if not names:
        die("no image files found in " + label + " directory: " + d)
    return names


def stem(n):
    return os.path.splitext(n)[0]


def ext_census(names):
    c = {}
    for n in names:
        e = os.path.splitext(n)[1].lower()
        c[e] = c.get(e, 0) + 1
    return c


def pct(num, den):
    if den == 0:
        return 0.0
    return 100.0 * float(num) / float(den)


def even_sample(items, k):
    """Deterministic, evenly spaced across the sorted list (not just the first k)."""
    n = len(items)
    if n <= k:
        return list(items)
    idx = np.linspace(0, n - 1, k).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [items[i] for i in idx]


def section(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--thermal", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--dim-check", type=int, default=0,
                    help="how many pairs to check dimensions on; 0 = all")
    ap.add_argument("--max-listed", type=int, default=40)
    args = ap.parse_args()

    section("DATASET: " + args.name)
    print("rgb dir     : " + os.path.abspath(args.rgb))
    print("thermal dir : " + os.path.abspath(args.thermal))

    rgb_names = list_images(args.rgb, args.name + "/rgb")
    th_names = list_images(args.thermal, args.name + "/thermal")

    # ---------------- 1. counts and pairing ----------------
    section("1. COUNTS AND PAIRING (by filename stem)")
    print("rgb image files     : " + str(len(rgb_names)))
    print("thermal image files : " + str(len(th_names)))
    print("rgb extensions      : " + str(ext_census(rgb_names)))
    print("thermal extensions  : " + str(ext_census(th_names)))

    rgb_by_stem = {}
    for n in rgb_names:
        rgb_by_stem.setdefault(stem(n), n)
    th_by_stem = {}
    for n in th_names:
        th_by_stem.setdefault(stem(n), n)

    paired_stems = sorted(set(rgb_by_stem.keys()) & set(th_by_stem.keys()))
    rgb_only = sorted(set(rgb_by_stem.keys()) - set(th_by_stem.keys()))
    th_only = sorted(set(th_by_stem.keys()) - set(rgb_by_stem.keys()))

    print("")
    print("matched pairs       : " + str(len(paired_stems)))
    print("rgb without thermal : " + str(len(rgb_only)))
    print("thermal without rgb : " + str(len(th_only)))

    if rgb_only:
        print("")
        print("unpaired RGB files:")
        for s in rgb_only[:args.max_listed]:
            print("  " + rgb_by_stem[s])
        if len(rgb_only) > args.max_listed:
            print("  ... and " + str(len(rgb_only) - args.max_listed) + " more")
    if th_only:
        print("")
        print("unpaired THERMAL files:")
        for s in th_only[:args.max_listed]:
            print("  " + th_by_stem[s])
        if len(th_only) > args.max_listed:
            print("  ... and " + str(len(th_only) - args.max_listed) + " more")
    if not rgb_only and not th_only:
        print("")
        print("no unpaired files in either modality")

    if not paired_stems:
        die("no RGB/thermal pairs matched by filename in " + args.name)

    # ---------------- 2. dimension agreement across modalities ----------------
    section("2. PAIRED DIMENSION AGREEMENT (bounding-box transfer)")
    if args.dim_check == 0:
        dim_stems = paired_stems
    else:
        dim_stems = even_sample(paired_stems, args.dim_check)
    print("pairs checked       : " + str(len(dim_stems)) + " of " + str(len(paired_stems)))
    mismatches = []
    rgb_sizes = {}
    th_sizes = {}
    for s in dim_stems:
        rp = os.path.join(args.rgb, rgb_by_stem[s])
        tp = os.path.join(args.thermal, th_by_stem[s])
        with Image.open(rp) as im:
            rs = im.size
        with Image.open(tp) as im:
            ts = im.size
        rgb_sizes[rs] = rgb_sizes.get(rs, 0) + 1
        th_sizes[ts] = th_sizes.get(ts, 0) + 1
        if rs != ts:
            mismatches.append((s, rs, ts))

    print("distinct rgb sizes     : " + str(rgb_sizes))
    print("distinct thermal sizes : " + str(th_sizes))
    print("dimension mismatches   : " + str(len(mismatches)) +
          "  (" + ("%.2f" % pct(len(mismatches), len(dim_stems))) + "% of checked)")
    for s, rs, ts in mismatches[:args.max_listed]:
        print("  " + s + "  rgb=" + str(rs) + "  thermal=" + str(ts))
    if len(mismatches) > args.max_listed:
        print("  ... and " + str(len(mismatches) - args.max_listed) + " more")

    # ---------------- 3. thermal sample format ----------------
    section("3. THERMAL SAMPLE: FORMAT, MODE, DTYPE, SHAPE")
    samp = even_sample(paired_stems, args.sample)
    print("sampled thermal images : " + str(len(samp)) +
          " (evenly spaced across " + str(len(paired_stems)) + " sorted pairs)")

    modes = {}
    fmts = {}
    dtypes = {}
    shapes = {}

    mins = []
    maxs = []
    means = []
    stds = []
    p_lo = []
    p_hi = []
    frac0 = []
    frac255 = []
    nlevels = []

    ch_maxdev = []
    ch_meandev = []
    n_exact_gray = 0
    n_near_gray = 0
    n_colour = 0
    NEAR_TOL = 4

    rows = []

    for s in samp:
        tp = os.path.join(args.thermal, th_by_stem[s])
        with Image.open(tp) as im:
            fmts[im.format] = fmts.get(im.format, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            a = np.array(im)

        dtypes[str(a.dtype)] = dtypes.get(str(a.dtype), 0) + 1
        shapes[str(a.shape)] = shapes.get(str(a.shape), 0) + 1

        if a.ndim == 3 and a.shape[2] >= 3:
            r = a[:, :, 0].astype(np.int16)
            g = a[:, :, 1].astype(np.int16)
            b = a[:, :, 2].astype(np.int16)
            d1 = np.abs(r - g)
            d2 = np.abs(g - b)
            d3 = np.abs(r - b)
            mx = int(max(d1.max(), d2.max(), d3.max()))
            mn = float((d1.mean() + d2.mean() + d3.mean()) / 3.0)
            ch_maxdev.append(mx)
            ch_meandev.append(mn)
            if mx == 0:
                n_exact_gray += 1
            elif mx <= NEAR_TOL:
                n_near_gray += 1
            else:
                n_colour += 1
            gray = a[:, :, 0]
        else:
            ch_maxdev.append(0)
            ch_meandev.append(0.0)
            n_exact_gray += 1
            gray = a

        gi = gray.astype(np.uint8)
        gf = gi.astype(np.float64)
        mn_v = int(gi.min())
        mx_v = int(gi.max())
        mins.append(mn_v)
        maxs.append(mx_v)
        means.append(float(gf.mean()))
        stds.append(float(gf.std()))
        lo, hi = np.percentile(gf, [0.5, 99.5])
        p_lo.append(float(lo))
        p_hi.append(float(hi))
        npix = gi.size
        frac0.append(float((gi == 0).sum()) / npix)
        frac255.append(float((gi == 255).sum()) / npix)
        nlevels.append(int(np.unique(gi).size))

        rows.append((s, mn_v, mx_v, mx_v - mn_v, float(gf.mean()), float(gf.std())))

    print("")
    print("PIL format census : " + str(fmts))
    print("PIL mode census   : " + str(modes))
    print("numpy dtype census: " + str(dtypes))
    print("array shape census: " + str(shapes))

    # ---------------- 4. channel structure ----------------
    section("4. CHANNEL STRUCTURE (grayscale-in-RGB vs baked-in colour palette)")
    n = len(samp)
    print("3 channels exactly identical (R==G==B)            : " +
          str(n_exact_gray) + "  (" + ("%.1f" % pct(n_exact_gray, n)) + "%)")
    print("near-identical (max channel delta <= " + str(NEAR_TOL) + ")           : " +
          str(n_near_gray) + "  (" + ("%.1f" % pct(n_near_gray, n)) + "%)")
    print("genuinely different channels (colour palette)     : " +
          str(n_colour) + "  (" + ("%.1f" % pct(n_colour, n)) + "%)")
    cm = np.array(ch_maxdev, dtype=np.float64)
    cd = np.array(ch_meandev, dtype=np.float64)
    print("")
    print("per-image MAX abs channel delta : min=" + str(int(cm.min())) +
          "  median=" + ("%.1f" % float(np.median(cm))) +
          "  mean=" + ("%.2f" % float(cm.mean())) +
          "  max=" + str(int(cm.max())))
    print("per-image MEAN abs channel delta: min=" + ("%.4f" % float(cd.min())) +
          "  median=" + ("%.4f" % float(np.median(cd))) +
          "  mean=" + ("%.4f" % float(cd.mean())) +
          "  max=" + ("%.4f" % float(cd.max())))

    # ---------------- 5. per-image range / rescaling test ----------------
    section("5. PER-IMAGE INTENSITY RANGE (the per-image-rescaling test)")
    mins_a = np.array(mins, dtype=np.float64)
    maxs_a = np.array(maxs, dtype=np.float64)
    span_a = maxs_a - mins_a
    mean_a = np.array(means, dtype=np.float64)
    std_a = np.array(stds, dtype=np.float64)
    lo_a = np.array(p_lo, dtype=np.float64)
    hi_a = np.array(p_hi, dtype=np.float64)
    f0_a = np.array(frac0, dtype=np.float64)
    f255_a = np.array(frac255, dtype=np.float64)
    lvl_a = np.array(nlevels, dtype=np.float64)

    def describe(label, arr):
        q = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100])
        print(label.ljust(22) +
              "min=" + ("%8.3f" % q[0]) +
              "  p5=" + ("%8.3f" % q[1]) +
              "  p25=" + ("%8.3f" % q[2]) +
              "  median=" + ("%8.3f" % q[3]) +
              "  p75=" + ("%8.3f" % q[4]) +
              "  p95=" + ("%8.3f" % q[5]) +
              "  max=" + ("%8.3f" % q[6]) +
              "  mean=" + ("%8.3f" % float(arr.mean())))

    print("distribution across the " + str(n) + " sampled thermal images:")
    print("")
    describe("per-image min", mins_a)
    describe("per-image max", maxs_a)
    describe("per-image span", span_a)
    describe("per-image mean", mean_a)
    describe("per-image std", std_a)
    describe("p0.5 (robust min)", lo_a)
    describe("p99.5 (robust max)", hi_a)
    describe("frac pixels == 0", f0_a)
    describe("frac pixels == 255", f255_a)
    describe("distinct gray levels", lvl_a)

    print("")
    print("--- full-range saturation counts (strict and loose bands) ---")
    for lo_t, hi_t in ((0, 255), (2, 253), (5, 250), (10, 245)):
        a = int((mins_a <= lo_t).sum())
        b = int((maxs_a >= hi_t).sum())
        both = int(((mins_a <= lo_t) & (maxs_a >= hi_t)).sum())
        print("min<=" + str(lo_t).rjust(2) + " : " + str(a).rjust(4) +
              " (" + ("%5.1f" % pct(a, n)) + "%)   " +
              "max>=" + str(hi_t).rjust(3) + " : " + str(b).rjust(4) +
              " (" + ("%5.1f" % pct(b, n)) + "%)   " +
              "BOTH : " + str(both).rjust(4) +
              " (" + ("%5.1f" % pct(both, n)) + "%)")

    print("")
    print("--- same test on robust percentiles (ignores a few stray pixels) ---")
    rb = int(((lo_a <= 5.0) & (hi_a >= 250.0)).sum())
    print("p0.5<=5   AND p99.5>=250 : " + str(rb) + " (" + ("%.1f" % pct(rb, n)) + "%)")
    rb2 = int(((lo_a <= 20.0) & (hi_a >= 235.0)).sum())
    print("p0.5<=20  AND p99.5>=235 : " + str(rb2) + " (" + ("%.1f" % pct(rb2, n)) + "%)")

    print("")
    print("--- first 30 sampled images, raw per-image numbers ---")
    print("name".ljust(18) + "min".rjust(5) + "max".rjust(6) + "span".rjust(6) +
          "mean".rjust(10) + "std".rjust(9))
    for r in rows[:30]:
        print(str(r[0]).ljust(18) + str(r[1]).rjust(5) + str(r[2]).rjust(6) +
              str(r[3]).rjust(6) + ("%10.3f" % r[4]) + ("%9.3f" % r[5]))

    section("DONE: " + args.name)


if __name__ == "__main__":
    main()
