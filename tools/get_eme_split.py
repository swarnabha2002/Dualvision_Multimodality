"""
Fetch and validate the EME (M3FD-zxSplit) scene-based train/test split.

Why a scene split matters: M3FD is captured as scenes -- many consecutive
frames from the same location, seconds apart. A random split puts near
identical images on both sides, inflates test accuracy, and means nothing.

This script:
  1. downloads train.txt / test.txt from the EME repo (or takes local copies)
  2. checks the stems actually match your 4200 local images
  3. checks train and test do not overlap
  4. reports class instance counts on each side
  5. independently detects scene boundaries from thermal statistics and
     checks whether the split respects them
  6. flags the 274 odd-resolution images

Step 5 is the one worth reading. It tells you whether to trust the split
rather than just assuming it is fine because it was published.

Usage:
    python tools/get_eme_split.py
    python tools/get_eme_split.py --train path/to/train.txt --test path/to/test.txt

Needs: numpy, Pillow. urllib for the download. No torch, no GPU.
"""

import os
import sys
import glob
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

M3FD = os.path.join("data", "M3FD")
VI = os.path.join(M3FD, "vi")
IR = os.path.join(M3FD, "ir")
LABELS = os.path.join(M3FD, "labels")
SPLIT_DIR = os.path.join(M3FD, "splits")

REPO_RAW = "https://raw.githubusercontent.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection/main/"

# The repo layout has moved before, so try several paths rather than one.
CANDIDATE_PATHS = [
    "dataset/m3fd-zxSplit/",
    "dataset/M3FD-zxSplit/",
    "datasets/m3fd-zxSplit/",
    "m3fd-zxSplit/",
]


# ------------------------------------------------------------------ fetching
def try_download(filename):
    for sub in CANDIDATE_PATHS:
        url = REPO_RAW + sub + filename
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                if r.status == 200:
                    text = r.read().decode("utf-8", errors="replace")
                    print("  got " + filename + " from " + sub)
                    return text
        except Exception:
            continue
    return None


def load_split(name, local_path):
    if local_path:
        if not os.path.exists(local_path):
            print("No such file: " + local_path)
            sys.exit(1)
        print("  reading local " + local_path)
        return open(local_path, encoding="utf-8", errors="replace").read()

    text = try_download(name)
    if text is None:
        print("")
        print("Could not download " + name + " from the EME repo.")
        print("The repo layout may have changed. Do this instead:")
        print("  1. open https://github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection")
        print("  2. find the m3fd-zxSplit folder and download train.txt and test.txt")
        print("  3. re-run with --train <path> --test <path>")
        sys.exit(1)
    return text


def parse_stems(text):
    """
    The files may hold bare stems, filenames with extensions, or full paths.
    Normalise everything to the bare stem.
    """
    stems = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace("\\", "/")
        base = line.split("/")[-1]
        stem = os.path.splitext(base)[0]
        stems.append(stem)
    return stems


# ------------------------------------------------------------------ local data
def local_stems(directory):
    out = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for p in glob.glob(os.path.join(directory, "*" + ext)):
            out.append(os.path.splitext(os.path.basename(p))[0])
    return sorted(set(out))


def check_match(split_stems, have, label):
    s = set(split_stems)
    missing = s - have
    print("  " + label + " entries : " + str(len(split_stems))
          + "  unique: " + str(len(s)))
    if len(s) != len(split_stems):
        print("  !! " + str(len(split_stems) - len(s)) + " duplicate lines")
    if missing:
        print("  !! " + str(len(missing)) + " stems in the split are NOT on disk")
        print("     e.g. " + str(sorted(missing)[:5]))
        print("     Usually a zero-padding or extension mismatch. Compare:")
        print("       split : " + sorted(s)[0])
        print("       disk  : " + sorted(have)[0])
    else:
        print("  all " + label + " stems found on disk")
    return s


# ------------------------------------------------------------------ classes
def class_counts(stems):
    counts = Counter()
    images_with = Counter()
    missing_xml = 0
    for stem in stems:
        path = os.path.join(LABELS, stem + ".xml")
        if not os.path.exists(path):
            missing_xml += 1
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            missing_xml += 1
            continue
        seen = set()
        for obj in root.findall("object"):
            el = obj.find("name")
            if el is None or el.text is None:
                continue
            cls = el.text.strip()
            counts[cls] += 1
            seen.add(cls)
        for c in seen:
            images_with[c] += 1
    return counts, images_with, missing_xml


def report_classes(train_stems, test_stems):
    tr_c, tr_i, tr_miss = class_counts(train_stems)
    te_c, te_i, te_miss = class_counts(test_stems)

    if tr_miss or te_miss:
        print("  missing/unparseable xml -- train " + str(tr_miss)
              + ", test " + str(te_miss))

    all_classes = sorted(set(tr_c) | set(te_c),
                         key=lambda c: -(tr_c[c] + te_c[c]))

    print("")
    print("  class          train inst   test inst   test share")
    print("  " + "-" * 52)
    for c in all_classes:
        total = tr_c[c] + te_c[c]
        share = (te_c[c] / total * 100) if total else 0.0
        flag = ""
        if total >= 200 and (share < 8 or share > 40):
            flag = "  <-- skewed"
        print("  " + c.ljust(14) + str(tr_c[c]).rjust(10)
              + str(te_c[c]).rjust(12) + (str(round(share, 1)) + "%").rjust(12)
              + flag)

    print("")
    print("  A class marked skewed has a very different train/test balance from")
    print("  the others. Not fatal, but report it -- per-class test numbers on a")
    print("  thin class will be noisy, and Lamp is one of your referent types.")


# ------------------------------------------------------------------ scenes
def thermal_mean(stem):
    path = os.path.join(IR, stem + ".png")
    if not os.path.exists(path):
        return None, None
    img = Image.open(path)
    size = img.size
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return float(arr.astype(np.float64).mean()), size


def detect_scenes(stems, jump_threshold=6.0):
    """
    Independent scene detection.

    Consecutive frames of one scene have near-identical mean thermal intensity.
    A jump between neighbours means the camera moved or conditions changed.
    Crude, but it does not depend on trusting anyone's published split.
    """
    print("")
    print("  scanning thermal means to find scene boundaries ...")
    means = OrderedDict()
    sizes = OrderedDict()
    for i, stem in enumerate(stems):
        m, sz = thermal_mean(stem)
        if m is None:
            continue
        means[stem] = m
        sizes[stem] = sz
        if (i + 1) % 500 == 0:
            print("    " + str(i + 1) + " / " + str(len(stems)))

    ordered = list(means.keys())
    boundaries = []
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        d = abs(means[cur] - means[prev])
        size_changed = sizes[cur] != sizes[prev]
        if d > jump_threshold or size_changed:
            boundaries.append(i)

    scene_of = {}
    scene_id = 0
    for i, stem in enumerate(ordered):
        if i in boundaries:
            scene_id += 1
        scene_of[stem] = scene_id

    sizes_per_scene = Counter(scene_of.values())
    print("    detected " + str(scene_id + 1) + " scenes")
    print("    frames per scene: min " + str(min(sizes_per_scene.values()))
          + ", median " + str(int(np.median(list(sizes_per_scene.values()))))
          + ", max " + str(max(sizes_per_scene.values())))
    return scene_of


def check_scene_purity(scene_of, train_set, test_set):
    per_scene = {}
    for stem, sid in scene_of.items():
        side = None
        if stem in train_set:
            side = "train"
        elif stem in test_set:
            side = "test"
        else:
            side = "unused"
        per_scene.setdefault(sid, Counter())[side] += 1

    impure = []
    for sid, c in per_scene.items():
        if c["train"] > 0 and c["test"] > 0:
            impure.append((sid, c["train"], c["test"]))

    print("")
    print("  scenes total            : " + str(len(per_scene)))
    print("  scenes split across both : " + str(len(impure)))

    if not impure:
        print("  >> Clean. No detected scene has frames on both sides.")
        print("     The split respects scene structure, so use it as is.")
    else:
        print("  >> " + str(len(impure)) + " scenes have frames in BOTH train and test.")
        print("     First few (scene id, train frames, test frames):")
        for sid, a, b in impure[:10]:
            print("       scene " + str(sid) + ": " + str(a) + " train, " + str(b) + " test")
        print("")
        print("     Two readings. Either my boundary detection is splitting one")
        print("     real scene in two (likely if the counts are small and the")
        print("     threshold is tight), or the published split does leak.")
        print("     Re-run with a larger --jump to test the first explanation")
        print("     before concluding the second.")

    unused = sum(1 for s in scene_of if s not in train_set and s not in test_set)
    if unused:
        print("")
        print("  " + str(unused) + " images are in neither train nor test.")
        print("  EME may have dropped them. Decide whether you want them back.")


def check_resolutions(stems):
    sizes = Counter()
    odd = []
    for stem in stems:
        path = os.path.join(IR, stem + ".png")
        if not os.path.exists(path):
            continue
        sz = Image.open(path).size
        sizes[sz] += 1
        if sz != (1024, 768):
            odd.append(stem)
    print("")
    print("  resolutions present:")
    for sz, n in sizes.most_common():
        print("    " + str(sz) + " : " + str(n))
    if odd:
        print("")
        print("  " + str(len(odd)) + " images are not 1024x768.")
        print("  Stem ranges: " + str(odd[0]) + " ... " + str(odd[-1]))
        print("  These are probably a separate capture session. Consider dropping")
        print("  them -- mixed resolutions complicate scene splitting and box")
        print("  scaling for little gain at this count.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", help="local train.txt instead of downloading")
    ap.add_argument("--test", help="local test.txt instead of downloading")
    ap.add_argument("--jump", type=float, default=6.0,
                    help="thermal-mean jump that marks a scene boundary")
    ap.add_argument("--skip-scenes", action="store_true",
                    help="skip the scene scan (it opens every thermal image)")
    args = ap.parse_args()

    print("Project root: " + os.getcwd())
    for d in (VI, IR, LABELS):
        if not os.path.isdir(d):
            print("Missing " + d + " -- run get_m3fd.py first.")
            sys.exit(1)

    print("")
    print("=" * 62)
    print("  1. FETCHING THE SPLIT")
    print("=" * 62)
    train_text = load_split("train.txt", args.train)
    test_text = load_split("test.txt", args.test)

    os.makedirs(SPLIT_DIR, exist_ok=True)
    open(os.path.join(SPLIT_DIR, "train.txt"), "w").write(train_text)
    open(os.path.join(SPLIT_DIR, "test.txt"), "w").write(test_text)
    print("  saved copies into " + SPLIT_DIR)

    train_stems = parse_stems(train_text)
    test_stems = parse_stems(test_text)

    print("")
    print("=" * 62)
    print("  2. DO THE STEMS MATCH YOUR FILES")
    print("=" * 62)
    have = set(local_stems(IR))
    print("  images on disk : " + str(len(have)))
    tr = check_match(train_stems, have, "train")
    te = check_match(test_stems, have, "test")

    overlap = tr & te
    print("")
    if overlap:
        print("  !! " + str(len(overlap)) + " stems appear in BOTH train and test.")
        print("     Do not use this split until that is resolved.")
    else:
        print("  no train/test overlap")

    covered = tr | te
    print("  coverage : " + str(len(covered)) + " of " + str(len(have))
          + " images (" + str(round(len(covered) / len(have) * 100, 1)) + "%)")

    print("")
    print("=" * 62)
    print("  3. CLASS DISTRIBUTION ACROSS THE SPLIT")
    print("=" * 62)
    report_classes(sorted(tr), sorted(te))

    print("")
    print("=" * 62)
    print("  4. RESOLUTIONS")
    print("=" * 62)
    check_resolutions(sorted(have))

    if not args.skip_scenes:
        print("")
        print("=" * 62)
        print("  5. DOES THE SPLIT RESPECT SCENE BOUNDARIES")
        print("=" * 62)
        scene_of = detect_scenes(sorted(have), jump_threshold=args.jump)
        check_scene_purity(scene_of, tr, te)

    print("")
    print("Done. If section 5 came back clean, adopt the split and move on to")
    print("building the measurement script. You still need your own dedup on")
    print("top -- a scene split stops leakage, it does not stop you making")
    print("forty near-identical questions from one parked car.")


if __name__ == "__main__":
    main()
