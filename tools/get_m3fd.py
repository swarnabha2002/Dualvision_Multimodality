"""
Download and set up M3FD locally.

M3FD is hosted on Google Drive via the TarDAL repo. There is no stable direct
URL, so you pass the link in:

    python tools/get_m3fd.py --url "https://drive.google.com/file/d/XXXX/view"

Get the link from https://github.com/dlut-dimt/TarDAL (README, M3FD section).
Look for M3FD_Detection -- that is the one with bounding boxes. There is also
a fusion-only release without annotations; that one is no use to you.

If you already downloaded the zip by hand, skip the download:

    python tools/get_m3fd.py --zip /path/to/M3FD_Detection.zip

What this does:
  1. downloads (or takes) the zip
  2. extracts it
  3. finds the rgb / thermal / annotation folders whatever they are named
  4. reorganises into data/M3FD/{vi,ir,labels}
  5. checks pairing and counts objects per class

Step 5 is the one you actually care about. It tells you how many vehicle and
lamp instances exist, which decides whether the benchmark is viable.

Needs: gdown (download only), Pillow. No torch, no GPU.
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse
import xml.etree.ElementTree as ET
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

DEST = os.path.join("data", "M3FD")
WORK = os.path.join("data", "_m3fd_raw")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Folder names used by different M3FD mirrors. Checked case-insensitively.
RGB_NAMES = ["vi", "vis", "visible", "rgb", "vi_images", "Vis"]
THR_NAMES = ["ir", "inf", "infrared", "thermal", "ti", "Ir"]
ANN_NAMES = ["labels", "label", "annotation", "annotations", "xml", "Annotation"]


# ------------------------------------------------------------------ download
def download(url, out_dir):
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Run:")
        print("    pip install gdown")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "M3FD_Detection.zip")

    if os.path.exists(target) and os.path.getsize(target) > 100 * 1024 * 1024:
        print("Zip already present and looks complete, skipping download:")
        print("  " + target)
        return target

    print("Downloading. This is several GB and will take a while.")
    print("If it stalls at 0%, the link is probably not public -- open it in a")
    print("browser first, accept any warning, then try again.")

    gdown.download(url=url, output=target, quiet=False, fuzzy=True)

    if not os.path.exists(target):
        print("")
        print("Download produced no file. Two common causes:")
        print("  - Google Drive quota exceeded for this file (wait, or use a mirror)")
        print("  - the link needs manual confirmation; download in a browser and")
        print("    re-run with --zip pointing at the file")
        sys.exit(1)

    size_gb = os.path.getsize(target) / (1024 ** 3)
    print("Downloaded " + str(round(size_gb, 2)) + " GB")
    return target


def extract(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    marker = os.path.join(out_dir, ".extracted")
    if os.path.exists(marker):
        print("Already extracted, skipping.")
        return out_dir

    print("Extracting " + os.path.basename(zip_path) + " ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        print("That file is not a valid zip. The download is probably an HTML")
        print("error page rather than the archive. Check its size and open it")
        print("in a text editor to confirm.")
        sys.exit(1)

    open(marker, "w").write("ok")
    print("Extracted to " + out_dir)
    return out_dir


# ------------------------------------------------------------------ discover
def find_folder(root, candidates):
    """Walk the extracted tree looking for a folder matching any candidate."""
    wanted = set(c.lower() for c in candidates)
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if d.lower() in wanted:
                full = os.path.join(dirpath, d)
                n = len(os.listdir(full))
                hits.append((n, full))
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


def list_images(directory):
    found = []
    for ext in IMAGE_EXTS:
        found.extend(glob.glob(os.path.join(directory, "*" + ext)))
        found.extend(glob.glob(os.path.join(directory, "*" + ext.upper())))
    return sorted(set(found))


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def organise(raw_root, dest):
    rgb_src = find_folder(raw_root, RGB_NAMES)
    thr_src = find_folder(raw_root, THR_NAMES)
    ann_src = find_folder(raw_root, ANN_NAMES)

    print("")
    print("Found in the archive:")
    print("  rgb         : " + str(rgb_src))
    print("  thermal     : " + str(thr_src))
    print("  annotations : " + str(ann_src))

    if rgb_src is None or thr_src is None:
        print("")
        print("Could not locate the image folders. Here is the tree so you can")
        print("point the script manually:")
        for dirpath, dirnames, filenames in os.walk(raw_root):
            depth = dirpath.replace(raw_root, "").count(os.sep)
            if depth > 3:
                continue
            print("  " * depth + os.path.basename(dirpath) + "/  ("
                  + str(len(filenames)) + " files)")
        sys.exit(1)

    if ann_src is None:
        print("")
        print("WARNING: no annotation folder found. You may have downloaded the")
        print("fusion-only release. Your benchmark needs bounding boxes, so get")
        print("M3FD_Detection instead.")

    mapping = [(rgb_src, os.path.join(dest, "vi")),
               (thr_src, os.path.join(dest, "ir"))]
    if ann_src is not None:
        mapping.append((ann_src, os.path.join(dest, "labels")))

    for src, dst in mapping:
        os.makedirs(dst, exist_ok=True)
        n = 0
        for name in os.listdir(src):
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if os.path.isfile(s) and not os.path.exists(d):
                shutil.copy2(s, d)
                n += 1
        print("  copied " + str(n) + " files into " + dst)

    return dest


# ------------------------------------------------------------------ verify
def check_pairs(dest):
    rgb = list_images(os.path.join(dest, "vi"))
    thr = list_images(os.path.join(dest, "ir"))

    rgb_stems = set(stem(p) for p in rgb)
    thr_stems = set(stem(p) for p in thr)
    matched = rgb_stems & thr_stems

    print("")
    print("Pairing")
    print("  rgb images     : " + str(len(rgb)))
    print("  thermal images : " + str(len(thr)))
    print("  matched pairs  : " + str(len(matched)))
    if rgb_stems - thr_stems:
        print("  rgb with no thermal : " + str(len(rgb_stems - thr_stems)))
    if thr_stems - rgb_stems:
        print("  thermal with no rgb : " + str(len(thr_stems - rgb_stems)))
    return matched


def count_classes(dest):
    """
    The number that decides whether your benchmark is viable.

    You need referents whose thermal state VARIES. Vehicles (engine on/off)
    and lamps (lit/unlit) qualify. People do not -- a person is always warm.
    """
    ann_dir = os.path.join(dest, "labels")
    if not os.path.isdir(ann_dir):
        print("")
        print("No labels folder -- skipping class counts.")
        return

    xmls = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    txts = sorted(glob.glob(os.path.join(ann_dir, "*.txt")))

    counts = Counter()
    images_with = Counter()
    areas = {}

    if xmls:
        print("")
        print("Annotation format: VOC XML (" + str(len(xmls)) + " files)")
        for path in xmls:
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            seen = set()
            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is None:
                    continue
                cls = name_el.text.strip()
                counts[cls] += 1
                seen.add(cls)
                bb = obj.find("bndbox")
                if bb is not None:
                    try:
                        w = float(bb.find("xmax").text) - float(bb.find("xmin").text)
                        h = float(bb.find("ymax").text) - float(bb.find("ymin").text)
                        areas.setdefault(cls, []).append(w * h)
                    except (AttributeError, TypeError, ValueError):
                        pass
            for c in seen:
                images_with[c] += 1
    elif txts:
        print("")
        print("Annotation format: YOLO txt (" + str(len(txts)) + " files)")
        print("Class ids only -- no names. M3FD order is usually:")
        print("  0 People  1 Car  2 Bus  3 Motorcycle  4 Lamp  5 Truck")
        for path in txts:
            seen = set()
            for line in open(path):
                parts = line.split()
                if not parts:
                    continue
                cls = parts[0]
                counts[cls] += 1
                seen.add(cls)
                if len(parts) >= 5:
                    try:
                        areas.setdefault(cls, []).append(
                            float(parts[3]) * float(parts[4]))
                    except ValueError:
                        pass
            for c in seen:
                images_with[c] += 1
    else:
        print("")
        print("Labels folder exists but holds no .xml or .txt files.")
        return

    print("")
    print("  class            instances   images   median box area")
    print("  " + "-" * 52)
    for cls, n in counts.most_common():
        a = areas.get(cls, [])
        if a:
            a = sorted(a)
            med = a[len(a) // 2]
            med_s = str(int(med)) if med > 10 else str(round(med, 4))
        else:
            med_s = "n/a"
        print("  " + cls.ljust(16) + str(n).rjust(9)
              + str(images_with[cls]).rjust(9) + med_s.rjust(18))

    print("")
    print("Read it like this:")
    print("  Vehicles and lamps are your usable referents -- their thermal state")
    print("  varies independently of what they are. People are not, but keep them")
    print("  as control questions where the answer never changes.")
    print("")
    print("  Median box area matters too. If vehicle boxes are small, you cannot")
    print("  measure a bonnet separately from a roof, and the whole task collapses")
    print("  to 'is this blob warm'. Open some images and look before committing.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Google Drive link to M3FD_Detection.zip")
    ap.add_argument("--zip", help="path to an already-downloaded zip")
    ap.add_argument("--keep-raw", action="store_true",
                    help="do not delete the extracted staging folder")
    args = ap.parse_args()

    if not args.url and not args.zip:
        ap.error("give either --url or --zip")

    print("Project root: " + os.getcwd())

    if args.zip:
        zip_path = args.zip
        if not os.path.exists(zip_path):
            print("No such file: " + zip_path)
            sys.exit(1)
    else:
        zip_path = download(args.url, WORK)

    extract(zip_path, WORK)
    organise(WORK, DEST)
    check_pairs(DEST)
    count_classes(DEST)

    print("")
    print("Next: run your verification script on this.")
    print("    python tools/step1_verify.py --name M3FD \\")
    print("        --rgb data/M3FD/vi --thermal data/M3FD/ir --sample 300")

    if not args.keep_raw:
        print("")
        print("Staging folder left at " + WORK + " (several GB).")
        print("Delete it once the checks pass.")


if __name__ == "__main__":
    main()
