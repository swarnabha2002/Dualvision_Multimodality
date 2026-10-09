"""
Step B of the zero-shot pilot: does relative thermal encoding survive a
sensor change when absolute encoding does not?

THE EXPERIMENT

For every question we run the model several times, changing only how the
thermal image is prepared:

  pipeline:   raw thermal  ->  perturb (simulate a different camera)  ->  encode  ->  model

  perturbations
    none          the image as shipped
    offset+20     add 20 to every pixel      (a warmer day, or a shifted sensor)
    offset-20     subtract 20
    gain0.8       reduce contrast by 20%     (a different calibration)
    gain1.2       increase contrast by 20%
    gamma0.7      a non-linear tone curve    (different 16->8 bit mapping)
    gamma1.4      the opposite curve

  Offset and gain cancel EXACTLY in the relative encoding -- that is
  algebra, not a result, and those rows should show a flip rate of zero.
  Treat them as a correctness check on the pipeline. The gamma rows are
  the real experiment, because nothing guarantees rel survives them.

  encodings
    abs   leave the values alone. This is what every current model does,
          including Thermo-VL, which replicates the single thermal channel
          three times and feeds it straight in.

    rel   subtract a reference and divide by a robust spread:

              out = 128 + SCALE * (x - reference) / spread

          A constant offset cancels in the subtraction. A gain change
          cancels in the division. So `rel` should be almost unaffected by
          the perturbations, and `abs` should not.

WHAT TO LOOK AT

Not raw accuracy. A frozen VLM has never seen a relative-encoded thermal
image, so `rel` may well score lower overall. The claim being tested is
STABILITY: how far accuracy falls when the sensor changes, and how often
the model flips its answer. Those are the two numbers in the summary.

No training. Inference only.

Needs: torch, transformers, numpy, Pillow, accelerate.
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

M3FD = os.path.join("data", "M3FD")
IR = os.path.join(M3FD, "ir")
VI = os.path.join(M3FD, "vi")
QUESTIONS = os.path.join(M3FD, "pilot_questions.json")
RESULTS = os.path.join(M3FD, "pilot_results.json")

SCALE = 45.0          # spread of the relative-encoded output around 128


# ------------------------------------------------------------------ imaging
def read_thermal(stem):
    """Thermal images are 3 identical channels; take one and work in float."""
    arr = np.array(Image.open(os.path.join(IR, stem + ".png")))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64)


def perturb(therm, kind):
    """
    Simulate a different camera or a different day.

    These are applied to the RAW values, before any encoding, because that
    is where a real sensor difference would appear.
    """
    if kind == "none":
        return therm
    if kind.startswith("offset"):
        delta = float(kind.replace("offset", ""))
        return therm + delta
    if kind.startswith("gamma"):
        g = float(kind.replace("gamma", ""))
        # A non-linear tone curve, the kind different cameras apply during
        # their own 16-bit to 8-bit conversion. This one does NOT cancel in
        # a subtract-and-divide, so it is the honest test of the method.
        x = np.clip(therm, 0, 255) / 255.0
        return np.power(x, g) * 255.0
    if kind.startswith("gain"):
        g = float(kind.replace("gain", ""))
        # Scale around the image's own mean so the picture does not simply
        # go uniformly brighter, which would be an offset, not a gain.
        return (therm - therm.mean()) * g + therm.mean()
    raise ValueError("unknown perturbation: " + kind)


def encode_abs(therm):
    """What everyone does now: clip to 8 bits and hand it over unchanged."""
    return np.clip(therm, 0, 255)


def encode_rel(therm, ref_pct=50.0):
    """
    Relative encoding, fixed-reference version.

    reference : a percentile of the image, standing in for neutral
                background. The median works because most of a street scene
                is background.
    spread    : median absolute deviation, a measure of contrast that is not
                thrown off by a few very hot pixels the way the standard
                deviation is.

    Dividing by the spread is what makes this survive a gain change.
    Subtracting the reference is what makes it survive an offset.
    """
    reference = np.percentile(therm, ref_pct)
    mad = np.median(np.abs(therm - reference))
    spread = max(float(therm.std()), 1.0)            # never divide by ~zero
    out = 128.0 + SCALE * (therm - reference) / spread
    return np.clip(out, 0, 255)


def to_image(arr):
    """Single channel float -> 3-channel uint8 PIL image, as VLMs expect."""
    a = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([a, a, a], axis=-1))


# ------------------------------------------------------------------ model
class VLM:
    """
    Thin wrapper so the rest of the script does not care which model is used.

    For anyone new to this: the processor turns images and text into the
    numeric tensors the model wants, the model produces token ids, and the
    processor turns those back into text. We ask for very few new tokens
    because the answer is one word.
    """

    def __init__(self, model_id, dtype="bfloat16"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        print("loading " + model_id + " ...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("loaded in " + str(round(time.time() - t0, 1)) + "s")
        dev = next(self.model.parameters()).device
        print("device: " + str(dev))

    def ask(self, images, question):
        torch = self.torch
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question
                        + " Answer with exactly one word: Yes or No."})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=6, do_sample=False)

        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def parse_yes_no(text):
    low = text.lower()
    has_yes = "yes" in low
    has_no = "no" in low
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return None          # unparseable; counted separately, never as correct


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200,
                    help="how many questions to run (each runs under every condition)")
    ap.add_argument("--modality", default="thermal", choices=["thermal", "both"],
                    help="thermal only isolates the effect; both is closer to real use")
    ap.add_argument("--ref-pct", type=float, default=50.0,
                    help="percentile used as the neutral reference for rel encoding")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resize", type=int, default=512,
                    help="longest side; smaller is much faster")
    args = ap.parse_args()

    print("Project root: " + os.getcwd())

    if not os.path.exists(QUESTIONS):
        print("No " + QUESTIONS + " -- run make_pilot_questions.py first.")
        sys.exit(1)

    items = json.load(open(QUESTIONS))[:args.limit]
    print("questions: " + str(len(items)))
    print("answer balance: " + str(dict(Counter(x["answer"] for x in items))))

    perturbations = ["none", "offset+20", "offset-20", "gain0.8", "gain1.2",
                     "gamma0.7", "gamma1.4"]
    encodings = ["abs", "rel"]
    total_calls = len(items) * len(perturbations) * len(encodings)
    print("model calls: " + str(total_calls))
    print("")

    vlm = VLM(args.model, dtype=args.dtype)

    records = []
    t0 = time.time()
    done = 0

    for item in items:
        therm_raw = read_thermal(item["stem"])
        rgb = None
        if args.modality == "both":
            rgb = Image.open(os.path.join(VI, item["stem"] + ".png")).convert("RGB")
            if args.resize:
                rgb.thumbnail((args.resize, args.resize))

        for pert in perturbations:
            perturbed = perturb(therm_raw, pert)
            for enc in encodings:
                if enc == "abs":
                    arr = encode_abs(perturbed)
                else:
                    arr = encode_rel(perturbed, ref_pct=args.ref_pct)

                img = to_image(arr)
                if args.resize:
                    img.thumbnail((args.resize, args.resize))

                images = [rgb, img] if rgb is not None else [img]
                raw = vlm.ask(images, item["question"])
                pred = parse_yes_no(raw)

                records.append({
                    "stem": item["stem"],
                    "question": item["question"],
                    "gold": item["answer"],
                    "kind": item["kind"],
                    "perturbation": pert,
                    "encoding": enc,
                    "raw_output": raw,
                    "pred": pred,
                    "correct": (pred == item["answer"]),
                })

                done += 1
                if done % 50 == 0:
                    rate = done / (time.time() - t0)
                    left = (total_calls - done) / max(rate, 1e-6)
                    print("  " + str(done) + " / " + str(total_calls)
                          + "   " + str(round(rate, 2)) + " calls/s"
                          + "   eta " + str(int(left // 60)) + " min")

    with open(RESULTS, "w") as f:
        json.dump({"args": vars(args), "records": records}, f)
    print("")
    print("wrote " + RESULTS)

    # ---------------------------------------------------------- summary
    acc = defaultdict(list)
    unparsed = Counter()
    for r in records:
        acc[(r["encoding"], r["perturbation"])].append(1.0 if r["correct"] else 0.0)
        if r["pred"] is None:
            unparsed[r["encoding"]] += 1

    print("")
    print("=" * 62)
    print("  ACCURACY BY ENCODING AND PERTURBATION")
    print("=" * 62)
    header = "  perturbation".ljust(18)
    for enc in encodings:
        header += enc.rjust(10)
    print(header)
    print("  " + "-" * 40)
    for pert in perturbations:
        line = ("  " + pert).ljust(18)
        for enc in encodings:
            v = acc[(enc, pert)]
            line += (str(round(100 * float(np.mean(v)), 1)) + "%").rjust(10)
        print(line)

    print("")
    print("=" * 62)
    print("  THE NUMBER THAT MATTERS: DROP FROM UNPERTURBED")
    print("=" * 62)
    for enc in encodings:
        base = float(np.mean(acc[(enc, "none")]))
        worst = min(float(np.mean(acc[(enc, p)])) for p in perturbations)
        mean_pert = float(np.mean([np.mean(acc[(enc, p)])
                                   for p in perturbations if p != "none"]))
        print("  " + enc
              + "   clean " + (str(round(100 * base, 1)) + "%").rjust(7)
              + "   mean perturbed " + (str(round(100 * mean_pert, 1)) + "%").rjust(7)
              + "   worst " + (str(round(100 * worst, 1)) + "%").rjust(7)
              + "   drop " + (str(round(100 * (base - worst), 1)) + " pts").rjust(9))

    # ---- flip rate: does the answer change when only the sensor changes?
    print("")
    print("=" * 62)
    print("  ANSWER FLIP RATE UNDER PERTURBATION")
    print("=" * 62)
    print("  (how often the answer changes when only the sensor changed;")
    print("   needs no ground truth, so it is the cleanest signal here)")
    print("")
    by_key = {}
    for r in records:
        by_key[(r["stem"], r["question"], r["encoding"], r["perturbation"])] = r["pred"]
    for enc in encodings:
        flips, total = 0, 0
        for item in items:
            base = by_key.get((item["stem"], item["question"], enc, "none"))
            for pert in perturbations:
                if pert == "none":
                    continue
                other = by_key.get((item["stem"], item["question"], enc, pert))
                total += 1
                if base != other:
                    flips += 1
        print("  " + enc + "   " + str(round(100 * flips / max(total, 1), 1))
              + "%  (" + str(flips) + " of " + str(total) + ")")

    if sum(unparsed.values()):
        print("")
        print("  unparseable outputs (counted as wrong): " + str(dict(unparsed)))

    yes_frac = np.mean([1.0 if x["answer"] == "Yes" else 0.0 for x in items])
    print("")
    print("  always-Yes baseline: " + str(round(100 * float(yes_frac), 1)) + "%")
    print("")
    print("  Offset and gain are exactly invariant for rel by construction: if")
    print("  those rows are not 0.0% flips, there is a bug in the pipeline.")
    print("  The gamma rows are the real test.")
    print("")
    print("  Read it like this: if rel's drop and flip rate are clearly lower")
    print("  than abs's, the idea works and the trained version is worth")
    print("  building. If they are the same, tell me before writing more code.")


if __name__ == "__main__":
    main()
