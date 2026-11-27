"""从已完成的 partial JSON/NPZ 重建七家族最终统计，不重复训练。"""

import json
import os
import sys

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_family_p0 as external  # noqa: E402


def main():
    partial = os.path.join(external.OUT, "dart_external_family_partial.json")
    results = json.load(open(partial))
    for target in external.TARGETS:
        assert set(results["folds"][target]["seeds"]) == set(map(str, external.SEEDS))
    results["summary"] = {feature: external.paired_summary(results, feature)
                          for feature in external.FEATURES}
    results["coverage_summary"] = external.coverage_summary(results)
    out = os.path.join(external.OUT, "dart_external_family.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(json.dumps(results["coverage_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
