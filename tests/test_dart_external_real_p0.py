import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

from analysis import dart_external_real_p0 as external


def test_subset_preserves_imagenet_class_indices(tmp_path):
    tar_path = tmp_path / "shard.tar"
    labels = (705, 0, 999)
    with tarfile.open(tar_path, "w") as archive:
        for i, label in enumerate(labels):
            cls = str(label).encode()
            info = tarfile.TarInfo(f"s{i}.cls")
            info.size = len(cls)
            archive.addfile(info, io.BytesIO(cls))
            buffer = io.BytesIO()
            Image.new("RGB", (2, 2), color=(i, 0, 0)).save(buffer, format="WEBP")
            info = tarfile.TarInfo(f"s{i}.webp")
            info.size = buffer.tell()
            buffer.seek(0)
            archive.addfile(info, buffer)
    root = tmp_path / "subset"
    manifest = external.prepare_subset(str(tar_path), str(root), n=2)
    dataset = ImageFolder(root, allow_empty=True)
    assert len(dataset) == len(manifest) == 2
    assert len(dataset.classes) == 1000
    for path, label in dataset.samples:
        assert int(Path(path).parent.name) == label
    assert {row["label"] for row in manifest} == {label for _, label in dataset.samples}


def test_external_shards_restore_positions(monkeypatch):
    monkeypatch.setattr(external, "SHARD_COUNT", 2)
    monkeypatch.setattr(external, "SHARD_INDEX", 0)
    even = external.positions()
    monkeypatch.setattr(external, "SHARD_INDEX", 1)
    odd = external.positions()
    restored = np.empty(300, dtype=np.int64)
    restored[::2], restored[1::2] = even, odd
    assert np.array_equal(restored, np.arange(300))
