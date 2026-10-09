"""
Step A of the zero-shot pilot: build yes/no questions from M3FD boxes.

No generator model is involved. Every answer comes straight from the
annotation file, so the ground truth is exact by construction.

Two question types, both answerable from the thermal image alone:

  presence  "Is there at least one car in this image?"
  counting  "Are there more than two people in this image?"

Balanced: each image contributes one question whose answer is Yes and one
whose answer is No, so a model that always says Yes scores exactly 50%.

Only objects large enough to be visible are counted. A 20-pixel car in the
distance is not fairly answerable and would just add noise.

Output: data/M3FD/pilot_questions.json

CPU only. Needs nothing but the standard library.
"""

import os
import sys
import glob
import json
import random
import argparse
import xml.etree.ElementTree as ET
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

M3FD = os.path.join("data", "M3FD")
LABELS = os.path.join(M3FD, "labels")
IR = os.path.join(M3FD, "ir")
OUT = os.path.join(M3FD, "pilot_questions.json")

# Lamp is left out: too small to see reliably in a downscaled thermal image.
CLASSES = ["People", "Car", "Bus", "Truck", "Motorcycle"]

# Plural forms for readable questions.
PLURAL = {
    "People": "people",
    "Car": "cars",
    "Bus": "buses",
    "Truck": "trucks",
    "Motorcycle": "motorcycles",
}
SINGULAR = {
    "People": "person",
    "Car": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Motorcycle": "motorcycle",
}

MIN_AREA = 900        # below this an object is too small to judge fairly
STANDARD_SIZE = (1024, 768)


def load_counts(stem):
    """Count visible objects per class in one annotation file."""
    path = os.path.join(LABELS, stem + ".xml")
    if not os.path.exists(path):
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    counts = Counter()
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if name_el is None or bb is None:
            continue
        cls = name_el.text.strip()
        if cls not in CLASSES:
            continue
        try:
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue
        if (x2 - x1) * (y2 - y1) >= MIN_AREA:
            counts[cls] += 1
    return counts


def presence_question(cls, answer):
    return ("Is there at least one " + SINGULAR[cls] + " in this image?", answer)


def counting_question(cls, n, answer):
    word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))
    return ("Are there more than " + word + " " + PLURAL[cls] + " in this image?", answer)


def build_for_image(stem, counts, rng):
    """
    Produce one Yes item and one No item for this image, if possible.
    Returns a list of 0, 1 or 2 question dicts.
    """
    present = [c for c in CLASSES if counts[c] > 0]
    absent = [c for c in CLASSES if counts[c] == 0]
    items = []

    # --- a question whose answer is Yes
    yes_options = []
    for c in present:
        yes_options.append(("presence",) + presence_question(c, "Yes"))
    for c in present:
        if counts[c] >= 2:
            # "more than n" with n strictly below the true count
            n = rng.randint(1, min(counts[c] - 1, 4))
            yes_options.append(("counting",) + counting_question(c, n, "Yes"))
    if yes_options:
        kind, q, a = rng.choice(yes_options)
        items.append({"stem": stem, "kind": kind, "question": q, "answer": a})

    # --- a question whose answer is No
    no_options = []
    for c in absent:
        no_options.append(("presence",) + presence_question(c, "No"))
    for c in present:
        # "more than n" with n at or above the true count
        n = counts[c] + rng.randint(0, 2)
        if n <= 5:
            no_options.append(("counting",) + counting_question(c, n, "No"))
    if no_options:
        kind, q, a = rng.choice(no_options)
        items.append({"stem": stem, "kind": kind, "question": q, "answer": a})

    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=300,
                    help="how many images to draw questions from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["train", "test", "all"],
                    help="which split to draw from; test keeps the pilot honest")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Project root: " + os.getcwd())

    if not os.path.isdir(LABELS):
        print("Missing " + LABELS + " -- run get_m3fd.py first.")
        sys.exit(1)

    # --- which stems are eligible
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(LABELS, "*.xml")))
    print("annotation files : " + str(len(stems)))

    if args.split != "all":
        split_file = os.path.join(M3FD, "splits", args.split + ".txt")
        if os.path.exists(split_file):
            keep = set()
            for line in open(split_file, encoding="utf-8", errors="replace"):
                line = line.strip().replace("\\", "/")
                if line:
                    keep.add(os.path.splitext(line.split("/")[-1])[0])
            stems = [s for s in stems if s in keep]
            print("after " + args.split + " split : " + str(len(stems)))
        else:
            print("No split file at " + split_file + " -- using all images.")

    # --- drop odd resolutions, they complicate nothing here but keep it clean
    from PIL import Image
    eligible = []
    for s in stems:
        p = os.path.join(IR, s + ".png")
        if os.path.exists(p) and Image.open(p).size == STANDARD_SIZE:
            eligible.append(s)
    print("standard resolution : " + str(len(eligible)))

    rng.shuffle(eligible)

    # --- build questions
    items = []
    used_images = 0
    skipped = 0
    for stem in eligible:
        if used_images >= args.images:
            break
        counts = load_counts(stem)
        if counts is None or sum(counts.values()) == 0:
            skipped += 1
            continue
        new = build_for_image(stem, counts, rng)
        if len(new) < 2:
            skipped += 1
            continue
        items.extend(new)
        used_images += 1

    # --- verification
    print("")
    print("images used   : " + str(used_images))
    print("images skipped: " + str(skipped) + " (no objects, or could not make both answers)")
    print("questions     : " + str(len(items)))

    ans = Counter(x["answer"] for x in items)
    kinds = Counter(x["kind"] for x in items)
    print("answers       : " + str(dict(ans)))
    print("kinds         : " + str(dict(kinds)))

    if ans["Yes"] == 0 or ans["No"] == 0:
        print("!! unbalanced -- a constant answer would score well. Fix before running.")
        sys.exit(1)
    balance = ans["Yes"] / (ans["Yes"] + ans["No"])
    print("yes fraction  : " + str(round(balance, 3)))
    if abs(balance - 0.5) > 0.05:
        print("   (a bit off 50/50; the always-Yes baseline is reported in the results anyway)")

    print("")
    print("five examples:")
    for x in items[:5]:
        print("  [" + x["answer"] + "] " + x["stem"] + "  " + x["question"])

    with open(OUT, "w") as f:
        json.dump(items, f, indent=1)
    print("")
    print("wrote " + OUT)
    print("Next: python tools/run_pilot.py --limit 200")


if __name__ == "__main__":
    main()
