"""Manifest 与 audit 单元测试。

构造临时小目录, 验证:
  - manifest 行数/字段/generator 元数据正确
  - audit 能发现注入的跨 split 精确重复与错误 generator 标签
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.audit import audit, load_manifest  # noqa: E402
from datasets.manifest import build_rows  # noqa: E402


class _Args:
    check_split_overlap = True
    check_exact_duplicates = True
    check_near_duplicates = False
    check_labels = True
    check_generator_metadata = True
    check_decode = True
    phash_threshold = 8


def _img(path, value):
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def _make_tree(tmp_path):
    root = tmp_path / "ARForensics"
    for split in ("train", "val"):
        for cls in ("0_real", "1_fake"):
            (root / split / cls).mkdir(parents=True)
    _img(str(root / "train" / "0_real" / "a.png"), 10)
    _img(str(root / "train" / "1_fake" / "b.png"), 20)
    _img(str(root / "val" / "0_real" / "c.png"), 30)
    _img(str(root / "val" / "1_fake" / "d.png"), 40)
    # 注入: val 中与 train 完全相同的文件 (跨 split 精确重复)
    _img(str(root / "val" / "0_real" / "a_dup.png"), 10)
    for gen in ("LlamaGen", "VAR"):
        for cls in ("0_real", "1_fake"):
            (root / "test" / gen / cls).mkdir(parents=True)
        _img(str(root / "test" / gen / "0_real" / "r.png"), 50)
        _img(str(root / "test" / gen / "1_fake" / "f.png"), 60)
    return root


def test_manifest_rows(tmp_path):
    root = _make_tree(tmp_path)
    rows = list(build_rows(str(root), "arforensics", with_hash=True))
    assert len(rows) == 4 + 1 + 4  # train/val 4 + dup 1 + test 4
    for r in rows:
        assert set(r.keys()) == {
            "sample_id", "path", "sha256", "phash", "label", "generator",
            "generator_family", "source", "split", "original_split", "dataset",
        }
        if r["label"] == 0:
            assert r["generator"] == "real"
        else:
            assert r["generator"] != "real"
        assert r["sha256"] and r["phash"]
    train = [r for r in rows if r["split"] == "train"]
    assert all(r["dataset"] == "ARForensics-small-val-derived-v1" for r in train)
    test = [r for r in rows if r["split"] == "test"]
    assert {r["generator"] for r in test if r["label"] == 1} == {"LlamaGen", "VAR"}


def test_audit_finds_injected_duplicate_and_bad_metadata(tmp_path):
    root = _make_tree(tmp_path)
    rows = list(build_rows(str(root), "arforensics", with_hash=True))
    # 注入错误的 generator 元数据
    rows.append({**rows[0], "path": "train/1_fake/bad.png", "label": "1",
                 "generator": "real", "sample_id": "bad0"})
    manifest = tmp_path / "m.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    loaded = load_manifest(str(manifest))
    report = audit(loaded, str(root), _Args())
    assert report["checks"]["exact_duplicates"]["cross_split_groups"] >= 1
    assert report["checks"]["generator_metadata"]["invalid_count"] >= 1
    assert report["checks"]["decode"]["corrupt"] == 0
    # missing: bad.png 不存在, 应被报 missing
    assert report["checks"]["decode"]["missing"] >= 1
