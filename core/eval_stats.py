import os
import glob
import json
from collections import defaultdict

exp_dir = "/home/gpuuser6/suklav/dualvision/DualVision/runs/local_xattn_v1/20260804_061224/exps"
jsonl_files = sorted(glob.glob(os.path.join(exp_dir, "*.jsonl")))

if not jsonl_files:
    print(f"No .jsonl prediction files found in {exp_dir}")
    exit(1)

stats = {}
category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
total_correct_all = 0
total_samples_all = 0

# Confusion matrix for binary classification
global_tp = 0  # True Yes
global_fp = 0  # Pred Yes, Label No
global_tn = 0  # True No
global_fn = 0  # Pred No, Label Yes

for filepath in jsonl_files:
    cond_name = os.path.basename(filepath).replace(".jsonl", "")
    
    # Categorize using substring matching on filenames
    if "blur" in cond_name:
        cat = "Blur"
    elif "bright" in cond_name or "dark" in cond_name:
        cat = "Darkness/Brightness"
    elif "fog" in cond_name:
        cat = "Fog"
    else:
        cat = "Clean / Baseline"

    correct = 0
    total = 0

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Extract prediction and ground truth using correct keys
            pred_raw = str(data.get("generated_answer", "")).strip().lower()
            label_raw = str(data.get("true_answer", "")).strip().lower()

            if not label_raw:
                continue

            # Prefix match check (e.g., "no, the sky..." starts with "no")
            is_correct = pred_raw.startswith(label_raw)
            
            if is_correct:
                correct += 1
            total += 1

            # Confusion Matrix
            pred_is_yes = pred_raw.startswith("yes")
            label_is_yes = label_raw.startswith("yes")

            if label_is_yes and pred_is_yes:
                global_tp += 1
            elif not label_is_yes and pred_is_yes:
                global_fp += 1
            elif not label_is_yes and not pred_is_yes:
                global_tn += 1
            elif label_is_yes and not pred_is_yes:
                global_fn += 1

    acc = (correct / total * 100) if total > 0 else 0.0
    stats[cond_name] = {"acc": acc, "correct": correct, "total": total}
    
    category_stats[cat]["correct"] += correct
    category_stats[cat]["total"] += total
    total_correct_all += correct
    total_samples_all += total

precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
overall_acc = (total_correct_all / total_samples_all * 100) if total_samples_all > 0 else 0

# Print Results
print("=" * 65)
print(f"        DUALVISION EVALUATION SUMMARY ({os.path.basename(os.path.dirname(exp_dir))})")
print("=" * 65)

print("\n1. OVERALL METRICS")
print("-" * 65)
print(f"Total Evaluated Samples : {total_samples_all}")
print(f"Overall Match Accuracy  : {overall_acc:.2f}% ({total_correct_all}/{total_samples_all})")
print(f"Precision ('Yes')       : {precision * 100:.2f}%")
print(f"Recall ('Yes')          : {recall * 100:.2f}%")
print(f"F1-Score                : {f1 * 100:.2f}%")

print("\n2. ACCURACY BY DEGRADATION GROUP")
print("-" * 65)
print(f"{'Group':<25} | {'Accuracy (%)':<12} | {'Correct / Total'}")
print("-" * 65)
for cat, data in category_stats.items():
    cat_acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
    print(f"{cat:<25} | {cat_acc:<12.2f} | {data['correct']}/{data['total']}")

print("\n3. PER-CONDITION DETAILED ACCURACY")
print("-" * 65)
print(f"{'Condition':<25} | {'Accuracy (%)':<12} | {'Correct / Total'}")
print("-" * 65)
for cond, data in stats.items():
    print(f"{cond:<25} | {data['acc']:<12.2f} | {data['correct']}/{data['total']}")
print("=" * 65)
