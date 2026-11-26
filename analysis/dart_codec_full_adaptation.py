"""完整压缩域适配：JPEG85 训练、选择、校准与测试均为压缩图。"""

import hashlib
import json
import os
import sys

import numpy as np
import torch

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_c2i_p0 as c2i  # noqa: E402
from analysis import dart_codec_p0 as codec  # noqa: E402
from analysis.dart_codec_calibration import pack_calibration, prada_score_calibration  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_p0 import normalize_data, predict, scale_tensor, train_dart  # noqa: E402
from prada.metrics import aeroblade  # noqa: E402

CODEC = codec.CODEC
SPLIT = c2i.fixed_test_split(0)
REF_IDX, SELECT_IDX, CAL_IDX, TEST_IDX, TRAIN_IDX, VAL_IDX, FAKE_TEST = SPLIT
REAL_POSITIONS = np.concatenate([REF_IDX, SELECT_IDX])
TRAIN_FAKE = {"var_d20": "var_d16", "var_d30": "var_d16",
              "llamagen_b256": "llamagen_b256",
              "llamagen_l256": "llamagen_b256", "rar_l": "rar_b"}
FAKE_POSITIONS = np.concatenate([TRAIN_IDX, VAL_IDX])
OUT = os.path.join(c2i.OUT, f"codec_{CODEC}_full_adaptation")
EXTRACT_ONLY = os.environ.get("DART_EXTRACT_ONLY", "0") == "1"


def cache(ds, model, positions):
    digest = hashlib.sha1(np.asarray(positions).tobytes()).hexdigest()[:8]
    return os.path.join(OUT, f"dataset={ds}_model={model}_n{len(positions)}_{digest}.pt")


@torch.no_grad()
def extract_one(model, ds, positions):
    path = cache(ds, model, positions)
    if os.path.exists(path):
        features = torch.load(path, map_location="cpu", weights_only=False)
        if all(key in features for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL",
                                           "quant_err", "rec_err")):
            return
    else:
        features = {"positions": np.asarray(positions)}
    wrapper = c2i.load_model_wrapper(model)
    if "cond_gt_llh_BL" not in features:
        rows = []
        for image, condition in codec.loader(ds, wrapper, positions=positions):
            image = image.to(wrapper.device, non_blocking=True)
            condition = condition.to(wrapper.device, non_blocking=True)
            gt = wrapper.get_gt_idx(image_B3HW=image)
            logits = wrapper.get_logits(gt_idx_BL=gt["gt_idx_BL"], condition_B=condition)
            llh = wrapper.get_gt_llh(logits_BLX=logits, gt_idx_BL=gt["gt_idx_BL"])
            rows.append({k: v.cpu() for k, v in llh.items()})
        for key in rows[0]:
            features[key] = torch.cat([row[key] for row in rows])
        if hasattr(wrapper, "scale_lengths"):
            features["scale_lengths"] = wrapper.scale_lengths
        codec.save_features(features, path)
    for key, batch_size in (("quant_err", c2i.cycle_batch_size(model)),
                            ("rec_err", c2i.rec_batch_size(model))):
        if key in features:
            continue
        values = []
        for image, _ in codec.loader(ds, wrapper, batch_size=batch_size, positions=positions):
            image = image.to(wrapper.device, non_blocking=True)
            rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
            if key == "quant_err":
                values.append(-rec["quant_err_BL"].cpu())
            else:
                values.append(-aeroblade(
                    img1=image, img2=rec["rec_B3HW"], net_type="vgg", reduction="none",
                    normalize=wrapper.range_after_transform == (0, 1)).cpu().reshape(-1))
        features[key] = torch.cat(values)
        codec.save_features(features, path)
    print(f"  full-adaptation {CODEC} {ds}/{model}", flush=True)


def extract(model):
    c2i.acquire_gpu()
    extract_one(model, "real", REAL_POSITIONS)
    extract_one(model, TRAIN_FAKE[model], FAKE_POSITIONS)
    torch.cuda.empty_cache()


def pack(path):
    feat = torch.load(path, map_location="cpu", weights_only=False)
    diff = (feat["cond_gt_llh_BL"] - feat["uncond_gt_llh_BL"]).float().numpy()
    lengths = feat.get("scale_lengths") or [diff.shape[1]]
    quant = feat["quant_err"].float().numpy()
    quant = np.stack([row.reshape(len(lengths), -1).mean(axis=1) for row in quant])
    return scale_tensor(diff, lengths, quant, feat["rec_err"].float().numpy())


def evaluate():
    real_train_select = {model: pack(cache("real", model, REAL_POSITIONS))
                         for model in codec.MODELS}
    compressed_test = {(ds, model): codec.pack_codec(ds, model)
                       for ds in codec.DATASETS for model in codec.MODELS}
    cal = {model: pack_calibration(model) for model in codec.MODELS}
    cal_prada = np.full(200, -np.inf, dtype=np.float32)
    dart_cal = np.full(200, -np.inf, dtype=np.float32)
    test_scores = {ds: np.full(len(codec.codec_positions(ds)), -np.inf, dtype=np.float32)
                   for ds in codec.DATASETS}
    test_prada = {ds: np.full(len(codec.codec_positions(ds)), -np.inf, dtype=np.float32)
                  for ds in codec.DATASETS}
    for model in codec.MODELS:
        gen = TRAIN_FAKE[model]
        fake = pack(cache(gen, model, FAKE_POSITIONS))
        normalized = normalize_data({"real": real_train_select[model], "fake": fake},
                                    np.arange(300))
        dart = train_dart(
            (np.concatenate([normalized["real"][:300], normalized["fake"][:100]]),
             np.array([0] * 300 + [1] * 100)),
            (np.concatenate([normalized["real"][300:500], normalized["fake"][100:150]]),
             np.array([0] * 200 + [1] * 50)), 0)
        real_scores = predict(dart, normalized["real"])
        median = np.median(real_scores[300:500])
        iqr = (np.percentile(real_scores[300:500], 75)
               - np.percentile(real_scores[300:500], 25) + 1e-6)
        ref = np.concatenate([real_train_select[model][i] for i in range(300)], axis=0)
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
        normalized_cal = np.asarray([(x - mu) / sd for x in cal[model]], dtype=object)
        dart_cal = np.maximum(dart_cal, (predict(dart, normalized_cal) - median) / iqr)
        cal_prada = np.maximum(cal_prada, prada_score_calibration(model))
        for ds in codec.DATASETS:
            normalized_test = np.asarray([(x - mu) / sd for x in compressed_test[ds, model]],
                                         dtype=object)
            test_scores[ds] = np.maximum(
                test_scores[ds], (predict(dart, normalized_test) - median) / iqr)
            test_prada[ds] = np.maximum(test_prada[ds], codec.codec_prada(ds, model))
    tau = conformal_tau(dart_cal)
    prada_tau = conformal_tau(cal_prada)
    results = {"config": {"codec": CODEC, "seed": 0,
                          "protocol": "compressed train/selection/calibration/test"},
               "generators": {}}
    real_rows = codec.position_rows("real", TEST_IDX)
    for gen in codec.FAKE_GENS:
        fake_rows = codec.position_rows(gen, FAKE_TEST)
        y = np.array([0] * 300 + [1] * 100)
        dart = np.concatenate([test_scores["real"][real_rows], test_scores[gen][fake_rows]])
        prada_score = np.concatenate([test_prada["real"][real_rows], test_prada[gen][fake_rows]])
        results["generators"][gen] = {"dart": c2i.metric(y, dart, 300, tau),
                                      "prada": c2i.metric(y, prada_score, 300, prada_tau)}
        np.savez(os.path.join(c2i.OUT, f"scores_dart_{CODEC}_compressed_full_{gen}.npz"),
                 dart=dart, prada=prada_score, n_real=300,
                 dart_cal=dart_cal, prada_cal=cal_prada,
                 dart_threshold=tau, prada_threshold=prada_tau)
    results["summary"] = {"macro_AUROC": float(np.mean([
        results["generators"][gen]["dart"]["AUROC"] for gen in codec.FAKE_GENS])),
        "FPR": float(results["generators"]["var_d16"]["dart"]["FPR"])}
    with open(os.path.join(c2i.OUT, f"dart_c2i_{CODEC}_compressed_full.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for model in codec.MODELS:
        extract(model)
    if not EXTRACT_ONLY:
        evaluate()


if __name__ == "__main__":
    main()
