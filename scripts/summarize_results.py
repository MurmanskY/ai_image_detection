import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "reproduction_summary.json")

METHODS = [
    "cnn_detection_arforensics",
    "frequency_arforensics",
    "npr_arforensics",
    "univfd_arforensics",
    "lgrad_arforensics",
    "dire_arforensics",
    "d3qe_lite_arforensics",
    "rigid_arforensics",
    "zed_arforensics",
]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.reader(f))


summary = {"standard_results": {}, "official_or_external_results": {}, "missing": []}
for name in METHODS:
    path = os.path.join(ROOT, "outputs", name, "results_test.json")
    if os.path.exists(path):
        summary["standard_results"][name] = read_json(path)
    else:
        summary["missing"].append(name)

for name in os.listdir(os.path.join(ROOT, "outputs")):
    if not name.startswith("dataprov") or not name.endswith(".csv"):
        continue
    path = os.path.join(ROOT, "outputs", name)
    summary["official_or_external_results"][name] = read_csv_rows(path)

d3qe_dir = os.path.join(ROOT, "third_party", "D3QE", "results", "D3QE")
if os.path.isdir(d3qe_dir):
    for name in sorted(os.listdir(d3qe_dir)):
        if name.endswith(".csv"):
            summary["official_or_external_results"][name[:-4]] = read_csv_rows(
                os.path.join(d3qe_dir, name)
            )
if "D3QE_arforensics_test" not in summary["official_or_external_results"]:
    summary["missing"].append("d3qe_official_arforensics")

for mode in ("c2i", "t2i"):
    root = os.path.join(ROOT, "outputs", f"prada_{mode}")
    if not os.path.isdir(root):
        summary["missing"].append(f"prada_{mode}")
        continue
    runs = sorted(
        os.path.join(root, name) for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )
    if runs:
        latest = runs[-1]
        result = {}
        for name in os.listdir(latest):
            if name.startswith("auroc_") and name.endswith(".csv"):
                result[name[:-4]] = read_csv_rows(os.path.join(latest, name))
        summary["official_or_external_results"][f"prada_{mode}"] = result
    else:
        summary["missing"].append(f"prada_{mode}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"结果汇总已保存到 {OUT}")
