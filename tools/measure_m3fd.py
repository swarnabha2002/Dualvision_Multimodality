"""
Step 2: turn M3FD boxes + thermal pixels into labelled instances.

The idea in one paragraph. You cannot tell which way a car faces from an
axis-aligned box, so you cannot locate its engine. Instead measure how
CONCENTRATED the heat is. A running car has one intense hot area against a
moderately warm body. A sun-heated car is uniformly warm. Same mean
temperature, different distribution -- and distribution needs no orientation.

Three features per object, all measured against a reference patch sampled
beside the object at the same height:

  elevation     median(object) - median(reference)
                how much warmer the object is than its surroundings

  concentration (p95(object) - median(reference)) / elevation
                how far the hottest part exceeds the typical part.
                engine ~ high, solar ~ near 1

  hot_y         vertical centroid of the hottest pixels, 0 = top of box
                solar heating lands on upper surfaces, engine heat lower

Labels that fall out:

  ACTIVE      warm, concentrated, not top-heavy        -> engine ran
  INACTIVE    not warm                                 -> did not
  SOLAR       warm, diffuse (height check optional)   -> sun, NOT running
  (rejected)  anything in between

SOLAR is the point of the whole exercise. A naive threshold calls those
active. They are not. That subset is what separates measuring from reasoning.

Outputs:
  data/M3FD/measurements.json    every object with its features
  data/M3FD/instances.json       deduped, labelled, split-assigned
  data/M3FD/sheets/*.png         contact sheets for manual checking

Run the sheets step and look at them. The labels are a hypothesis about what
the thermal pattern means, and nothing validates that except your eyes.

Needs numpy and Pillow. No torch, no GPU.
"""

import os
import sys
import glob
import json
import random
import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

M3FD = os.path.join("data", "M3FD")
VI = os.path.join(M3FD, "vi")
IR = os.path.join(M3FD, "ir")
LABELS = os.path.join(M3FD, "labels")
SPLITS = os.path.join(M3FD, "splits")
SHEETS = os.path.join(M3FD, "sheets")

VEHICLES = ["Car", "Bus", "Truck", "Motorcycle"]
WHOLE_BOX = ["Lamp"]          # too small for internal structure; box IS the measure
CONTROLS = ["People"]         # thermal state never varies; used as sanity checks

STANDARD_SIZE = (1024, 768)   # the 274 odd-resolution images are dropped

# Minimum box area to bother with. Below this there are too few pixels for
# the percentile statistics to mean anything.
MIN_AREA_VEHICLE = 1200
MIN_AREA_LAMP = 100


# ------------------------------------------------------------------ loading
def load_split():
    tr_path = os.path.join(SPLITS, "train.txt")
    te_path = os.path.join(SPLITS, "test.txt")
    if not os.path.exists(tr_path):
        print("No split files. Run get_eme_split.py first.")
        sys.exit(1)

    def stems(path):
        out = set()
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip().replace("\\", "/")
            if not line:
                continue
            out.add(os.path.splitext(line.split("/")[-1])[0])
        return out

    return stems(tr_path), stems(te_path)


def load_boxes(stem):
    path = os.path.join(LABELS, stem + ".xml")
    if not os.path.exists(path):
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []

    boxes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if name_el is None or bb is None:
            continue
        try:
            x1 = int(float(bb.find("xmin").text))
            y1 = int(float(bb.find("ymin").text))
            x2 = int(float(bb.find("xmax").text))
            y2 = int(float(bb.find("ymax").text))
        except (AttributeError, TypeError, ValueError):
            continue
        boxes.append({
            "cls": name_el.text.strip(),
            "box": [x1, y1, x2, y2],
        })
    return boxes


def read_thermal(stem):
    path = os.path.join(IR, stem + ".png")
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    if img.size != STANDARD_SIZE:
        return None
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64)


# ------------------------------------------------------------------ reference
def sample_reference(thermal, box, all_boxes, margin=0.5):
    """
    Take the reference from beside the object, in the same horizontal band.

    Same band means same depth and usually the same ground surface, which is
    what you want. Sampling from the top of the image gives you sky, which
    clips to zero and is thermally decoupled from anything on the ground.

    Returns (values, why_rejected). values is None if no clean patch exists.
    """
    H, W = thermal.shape
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    # Vertical band: the object's own rows, so same distance from camera.
    by1 = max(0, y1)
    by2 = min(H, y2)
    if by2 - by1 < 8:
        return None, "band too thin"

    pad = int(bw * margin)
    candidates = [
        (max(0, x1 - pad - bw), max(0, x1 - pad)),      # left of object
        (min(W, x2 + pad), min(W, x2 + pad + bw)),      # right of object
    ]

    # Build an occupancy mask so the reference never lands on another object.
    occupied = np.zeros((H, W), dtype=bool)
    for other in all_boxes:
        ox1, oy1, ox2, oy2 = other["box"]
        ox1 = max(0, ox1 - 4); oy1 = max(0, oy1 - 4)
        ox2 = min(W, ox2 + 4); oy2 = min(H, oy2 + 4)
        occupied[oy1:oy2, ox1:ox2] = True

    pooled = []
    for cx1, cx2 in candidates:
        if cx2 - cx1 < 8:
            continue
        patch = thermal[by1:by2, cx1:cx2]
        mask = ~occupied[by1:by2, cx1:cx2]
        vals = patch[mask]
        if vals.size < 100:
            continue
        # Reject clipped sky. A patch that is mostly zero carries no signal.
        if (vals == 0).mean() > 0.10:
            continue
        # Reject patches spanning several materials -- the median means little.
        if vals.std() > 45:
            continue
        pooled.append(vals)

    if not pooled:
        return None, "no clean reference patch"

    return np.concatenate(pooled), None


# ------------------------------------------------------------------ features
def measure(thermal, box, all_boxes, shrink=0.12):
    """
    Shrink the box before measuring. Annotation boxes include background at
    the corners, and for a car that background is road, which would drag the
    object median toward the reference and flatten your signal.
    """
    H, W = thermal.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    sx, sy = int(bw * shrink), int(bh * shrink)
    ix1 = max(0, x1 + sx); iy1 = max(0, y1 + sy)
    ix2 = min(W, x2 - sx); iy2 = min(H, y2 - sy)
    if ix2 - ix1 < 6 or iy2 - iy1 < 6:
        return None, "box too small after shrink"

    obj = thermal[iy1:iy2, ix1:ix2]
    if obj.size < 60:
        return None, "too few object pixels"

    ref, why = sample_reference(thermal, box, all_boxes)
    if ref is None:
        return None, why

    ref_med = float(np.median(ref))
    ref_std = float(ref.std())
    obj_med = float(np.median(obj))
    obj_p95 = float(np.percentile(obj, 95))

    elevation = obj_med - ref_med
    peak = obj_p95 - ref_med

    # Concentration is undefined when the object is not warmer at all, so
    # floor the denominator rather than letting it explode.
    concentration = peak / max(elevation, 1.0)

    # Vertical centroid of the hottest pixels, normalised 0 (top) to 1 (bottom).
    hot_thresh = np.percentile(obj, 85)
    hot_mask = obj >= hot_thresh
    if hot_mask.any():
        rows = np.nonzero(hot_mask)[0]
        hot_y = float(rows.mean() / max(obj.shape[0] - 1, 1))
    else:
        hot_y = 0.5

    # Same for horizontal, useful later for side-on vehicles.
    if hot_mask.any():
        cols = np.nonzero(hot_mask)[1]
        hot_x = float(cols.mean() / max(obj.shape[1] - 1, 1))
    else:
        hot_x = 0.5

    return {
        "elevation": round(elevation, 3),
        "concentration": round(concentration, 3),
        "hot_y": round(hot_y, 3),
        "hot_x": round(hot_x, 3),
        "obj_median": round(obj_med, 2),
        "obj_p95": round(obj_p95, 2),
        "ref_median": round(ref_med, 2),
        "ref_std": round(ref_std, 2),
        "obj_px": int(obj.size),
        "ref_px": int(ref.size),
    }, None


# ------------------------------------------------------------------ labelling
def assign_label(f, th):
    """
    Thresholds come from the measured distribution, not from physics, because
    the gray levels are not calibrated temperatures. Say so in the paper.
    """
    e = f["elevation"]
    c = f["concentration"]
    y = f["hot_y"]

    if e < th["e_lo"]:
        return "INACTIVE"

    if e >= th["e_hi"]:
        if c >= th["c_hi"] and y >= th["y_mid"]:
            return "ACTIVE"
        # Height requirement for SOLAR is off by default (solar_max_y = 1.01).
        # From street-level views the upper part of a car box is mostly glass,
        # which reflects cold sky, so almost no car reads warm on top.
        if c <= th["c_lo"] and y < th["solar_max_y"]:
            return "SOLAR"

    return None   # ambiguous, dropped


def derive_thresholds(records, args):
    """
    Elevation thresholds come from all vehicles. Concentration thresholds come
    ONLY from warm vehicles, because concentration is a ratio over elevation
    and is meaningless when the object is not warmer than its reference.

    Percentile thresholds are a tool for FINDING the boundary by inspecting
    the contact sheets. Once the sheets look right, freeze the printed values
    as absolute gray-level thresholds (valid because M3FD's thermal mapping is
    fixed) and report those in the paper.
    """
    el = np.array([r["feat"]["elevation"] for r in records])
    e_lo = float(np.percentile(el, args.e_lo_pct))
    e_hi = float(np.percentile(el, args.e_hi_pct))

    warm = [r for r in records if r["feat"]["elevation"] >= e_hi]
    print("  warm vehicles used for concentration thresholds : " + str(len(warm)))
    if len(warm) < 20:
        print("  WARNING: very few warm vehicles, concentration thresholds unreliable")
    if len(warm) == 0:
        warm = records

    co = np.array([r["feat"]["concentration"] for r in warm])
    qs = np.percentile(co, [5, 25, 50, 75, 95])
    print("  concentration among warm vehicles  p5 " + str(round(qs[0], 2))
          + "  p25 " + str(round(qs[1], 2)) + "  med " + str(round(qs[2], 2))
          + "  p75 " + str(round(qs[3], 2)) + "  p95 " + str(round(qs[4], 2)))

    return {
        "e_lo": e_lo,
        "e_hi": e_hi,
        "c_lo": float(np.percentile(co, args.c_lo_pct)),
        "c_hi": float(np.percentile(co, args.c_hi_pct)),
        "y_mid": args.y_mid,
    }


def print_distribution(records):
    el = np.array([r["feat"]["elevation"] for r in records])
    co = np.array([r["feat"]["concentration"] for r in records])
    hy = np.array([r["feat"]["hot_y"] for r in records])

    def line(name, a):
        qs = np.percentile(a, [5, 25, 50, 75, 95])
        print("  " + name.ljust(14)
              + "  p5 " + str(round(qs[0], 2)).rjust(8)
              + "  p25 " + str(round(qs[1], 2)).rjust(8)
              + "  med " + str(round(qs[2], 2)).rjust(8)
              + "  p75 " + str(round(qs[3], 2)).rjust(8)
              + "  p95 " + str(round(qs[4], 2)).rjust(8))

    print("")
    print("  feature distribution over " + str(len(records)) + " measured objects")
    line("elevation", el)
    line("concentration", co)
    line("hot_y", hy)
    print("")
    print("  If elevation has a long right tail, the warm objects are a minority")
    print("  and your ACTIVE class will be small. That is expected -- most parked")
    print("  cars in a street scene have cold engines.")


# ------------------------------------------------------------------ dedup
def scene_of_stem(stem, boundaries):
    idx = int(stem)
    lo = 0
    for i, b in enumerate(boundaries):
        if idx < b:
            return i
        lo = b
    return len(boundaries)


def detect_scene_boundaries(stems, jump=3.0):
    """
    Same scene detection as before but tighter, since over-splitting is the
    safe direction here: two fragments of one real scene simply become two
    dedup groups, which costs a little data and leaks nothing.
    """
    print("  scanning for scene boundaries ...")
    means = {}
    for i, s in enumerate(stems):
        t = read_thermal(s)
        if t is None:
            continue
        means[s] = float(t.mean())
        if (i + 1) % 800 == 0:
            print("    " + str(i + 1) + " / " + str(len(stems)))

    ordered = sorted(means.keys())
    scene = {}
    sid = 0
    prev = None
    for s in ordered:
        if prev is not None and abs(means[s] - means[prev]) > jump:
            sid += 1
        scene[s] = sid
        prev = s
    print("    " + str(sid + 1) + " scenes")
    return scene


def dedup(records, scene, pos_tol=0.06, size_tol=0.25):
    """
    One parked car across forty frames must not become forty questions.

    Group by (scene, class, roughly-same-position, roughly-same-size) and keep
    a single representative -- the one whose elevation is closest to the group
    median, so you keep a typical view rather than an outlier frame.
    """
    groups = defaultdict(list)
    for r in records:
        s = scene.get(r["stem"], -1)
        x1, y1, x2, y2 = r["box"]
        cx = (x1 + x2) / 2 / STANDARD_SIZE[0]
        cy = (y1 + y2) / 2 / STANDARD_SIZE[1]
        area = ((x2 - x1) * (y2 - y1)) ** 0.5
        # Label is part of the key: the same car in two different states is
        # two genuine instances, and merging them would silently relabel one.
        key = (s, r["cls"], r["label"],
               round(cx / pos_tol), round(cy / pos_tol),
               round(np.log(max(area, 1.0)) / size_tol))
        groups[key].append(r)

    kept = []
    for key, members in groups.items():
        if len(members) == 1:
            kept.append(members[0])
            continue
        els = [m["feat"]["elevation"] for m in members]
        med = float(np.median(els))
        best = min(members, key=lambda m: abs(m["feat"]["elevation"] - med))
        best["group_size"] = len(members)
        kept.append(best)

    print("")
    print("  before dedup : " + str(len(records)) + " objects")
    print("  after dedup  : " + str(len(kept)) + " instances")
    print("  collapsed    : " + str(round((1 - len(kept) / max(len(records), 1)) * 100, 1)) + "%")
    return kept


# ------------------------------------------------------------------ sheets
def contact_sheet(instances, label, out_path, n=24, crop=112):
    """
    RGB crop on top, thermal crop below, for n random instances of one label.
    Open these and check the labels make sense. Nothing else validates them.
    """
    picks = random.sample(instances, min(n, len(instances)))
    if not picks:
        return

    cols = 6
    rows = (len(picks) + cols - 1) // cols
    cell_w = crop
    cell_h = crop * 2 + 26
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)

    for i, r in enumerate(picks):
        cx, cy = i % cols, i // cols
        x1, y1, x2, y2 = r["box"]
        try:
            rgb = Image.open(os.path.join(VI, r["stem"] + ".png")).convert("RGB")
            thr = Image.open(os.path.join(IR, r["stem"] + ".png")).convert("RGB")
        except OSError:
            continue

        pad = int(max(x2 - x1, y2 - y1) * 0.15)
        bx = (max(0, x1 - pad), max(0, y1 - pad),
              min(rgb.width, x2 + pad), min(rgb.height, y2 + pad))

        rc = rgb.crop(bx).resize((crop, crop))
        tc = thr.crop(bx).resize((crop, crop))

        ox, oy = cx * cell_w, cy * cell_h
        sheet.paste(rc, (ox, oy))
        sheet.paste(tc, (ox, oy + crop))
        cap = ("e" + str(round(r["feat"]["elevation"], 1))
               + " c" + str(round(r["feat"]["concentration"], 1))
               + " y" + str(round(r["feat"]["hot_y"], 2)))
        draw.text((ox + 3, oy + crop * 2 + 4), cap, fill=(200, 205, 220))

    sheet.save(out_path)
    print("  wrote " + out_path + "  (" + str(len(picks)) + " of " + str(len(instances)) + ")")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-lo-pct", type=float, default=40.0)
    ap.add_argument("--e-hi-pct", type=float, default=70.0)
    ap.add_argument("--c-lo-pct", type=float, default=35.0)
    ap.add_argument("--c-hi-pct", type=float, default=65.0)
    ap.add_argument("--y-mid", type=float, default=0.40)
    ap.add_argument("--solar-max-y", type=float, default=1.01,
                    help="SOLAR needs hot_y below this; 1.01 disables the height check")
    ap.add_argument("--lamp-on", type=float, default=40.0,
                    help="absolute gray-level elevation above which a lamp counts as lit")
    ap.add_argument("--lamp-off", type=float, default=8.0,
                    help="absolute gray-level elevation below which a lamp counts as unlit")
    ap.add_argument("--jump", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="process only N images, for a quick trial")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Project root: " + os.getcwd())
    train_stems, test_stems = load_split()
    all_stems = sorted(train_stems | test_stems)
    if args.limit:
        all_stems = all_stems[:args.limit]

    print("")
    print("=" * 62)
    print("  MEASURING")
    print("=" * 62)

    records = []
    rejects = Counter()
    skipped_size = 0

    for i, stem in enumerate(all_stems):
        thermal = read_thermal(stem)
        if thermal is None:
            skipped_size += 1
            continue

        boxes = load_boxes(stem)
        if not boxes:
            continue

        for b in boxes:
            cls = b["cls"]
            if cls not in VEHICLES + WHOLE_BOX + CONTROLS:
                continue
            x1, y1, x2, y2 = b["box"]
            area = (x2 - x1) * (y2 - y1)
            floor = MIN_AREA_LAMP if cls in WHOLE_BOX else MIN_AREA_VEHICLE
            if area < floor:
                rejects["too small"] += 1
                continue

            shrink = 0.0 if cls in WHOLE_BOX else 0.12
            feat, why = measure(thermal, b["box"], boxes, shrink=shrink)
            if feat is None:
                rejects[why] += 1
                continue

            records.append({
                "stem": stem,
                "cls": cls,
                "box": b["box"],
                "feat": feat,
                "split": "train" if stem in train_stems else "test",
            })

        if (i + 1) % 500 == 0:
            print("  " + str(i + 1) + " / " + str(len(all_stems))
                  + "  measured " + str(len(records)))

    print("")
    print("  images skipped for odd resolution : " + str(skipped_size))
    print("  objects measured : " + str(len(records)))
    if rejects:
        print("  rejected:")
        for why, n in rejects.most_common():
            print("    " + str(why) + " : " + str(n))

    if not records:
        print("Nothing measured. Check paths.")
        sys.exit(1)

    print_distribution(records)

    # ---- labelling
    print("")
    print("=" * 62)
    print("  LABELLING")
    print("=" * 62)
    veh = [r for r in records if r["cls"] in VEHICLES]
    th = derive_thresholds(veh, args)
    print("  thresholds from the data:")
    th["solar_max_y"] = args.solar_max_y
    th["lamp_on"] = args.lamp_on
    th["lamp_off"] = args.lamp_off
    for k, v in th.items():
        print("    " + k.ljust(13) + str(round(v, 3)))

    for r in records:
        if r["cls"] in VEHICLES:
            r["label"] = assign_label(r["feat"], th)
        elif r["cls"] in WHOLE_BOX:
            e = r["feat"]["elevation"]
            r["label"] = "LAMP_ON" if e >= args.lamp_on else ("LAMP_OFF" if e < args.lamp_off else None)
        else:
            r["label"] = "CONTROL"

    labelled = [r for r in records if r["label"] is not None]
    print("")
    print("  label counts before dedup:")
    for lab, n in Counter(r["label"] for r in labelled).most_common():
        print("    " + lab.ljust(12) + str(n))
    print("")
    print("  dropped as ambiguous : " + str(len(records) - len(labelled)))

    # ---- dedup
    print("")
    print("=" * 62)
    print("  DEDUP BY SCENE")
    print("=" * 62)
    scene = detect_scene_boundaries(all_stems, jump=args.jump)
    instances = dedup(labelled, scene)

    print("")
    print("  final instances by label and split:")
    grid = defaultdict(Counter)
    for r in instances:
        grid[r["label"]][r["split"]] += 1
    for lab in sorted(grid):
        c = grid[lab]
        print("    " + lab.ljust(12) + " train " + str(c["train"]).rjust(6)
              + "   test " + str(c["test"]).rjust(6))

    # ---- write
    os.makedirs(SHEETS, exist_ok=True)
    with open(os.path.join(M3FD, "measurements.json"), "w") as f:
        json.dump({"thresholds": th, "records": records}, f)
    with open(os.path.join(M3FD, "instances.json"), "w") as f:
        json.dump({"thresholds": th, "instances": instances}, f)
    print("")
    print("  wrote measurements.json and instances.json")

    # ---- sheets
    print("")
    print("=" * 62)
    print("  CONTACT SHEETS -- GO AND LOOK AT THESE")
    print("=" * 62)
    by_label = defaultdict(list)
    for r in instances:
        by_label[r["label"]].append(r)
    for lab, items in by_label.items():
        contact_sheet(items, lab, os.path.join(SHEETS, lab + ".png"))
    for lab in ["ACTIVE", "SOLAR", "INACTIVE"]:
        if lab not in by_label:
            print("  NOTE: no " + lab + " instances, so no " + lab + ".png was written.")
            print("        Loosen the thresholds or tell me what the output shows.")

    print("")
    print("  Open data/M3FD/sheets/ACTIVE.png and SOLAR.png side by side.")
    print("  In ACTIVE you should see a localised bright patch on the vehicle.")
    print("  In SOLAR you should see broad even warmth on upper surfaces.")
    print("  If you cannot tell them apart by eye, the thresholds are wrong and")
    print("  no amount of training will fix it. Adjust the percentile arguments")
    print("  and re-run before going any further.")


if __name__ == "__main__":
    main()