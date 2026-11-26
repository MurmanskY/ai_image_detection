"""压缩 calibration 协议：原始训练/选择，压缩校准与测试（默认 JPEG85 seed 0）。"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_c2i_p0 as c2i  # noqa: E402
from analysis import dart_codec_p0 as codec  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_p0 import normalize_data, predict, scale_tensor, train_dart  # noqa: E402
from prada.metrics import aeroblade  # noqa: E402

CODEC = codec.CODEC
CAL_IDX = c2i.fixed_test_split(0)[2]
OUT = os.path.join(c2i.OUT, f"codec_{CODEC}_calibration")
EXTRACT_ONLY = os.environ.get("DART_EXTRACT_ONLY", "0") == "1"


def path(model):
    return os.path.join(OUT, f"real_model={model}.pt")


@torch.no_grad()
def extract(model):
    c2i.acquire_gpu()
    wrapper = c2i.load_model_wrapper(model)
    cache = path(model)
    features = torch.load(cache, map_location="cpu", weights_only=False) \
        if os.path.exists(cache) else {"positions": CAL_IDX}
    if "cond_gt_llh_BL" not in features:
        rows = []
        for image, condition in codec.loader("real", wrapper, positions=CAL_IDX):
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
        codec.save_features(features, cache)
    for key, batch_size in (("quant_err", c2i.cycle_batch_size(model)),
                            ("rec_err", c2i.rec_batch_size(model))):
        if key in features:
            continue
        values = []
        for image, _ in codec.loader("real", wrapper, batch_size=batch_size,
                                     positions=CAL_IDX):
            image = image.to(wrapper.device, non_blocking=True)
            rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
            if key == "quant_err":
                values.append(-rec["quant_err_BL"].cpu())
            else:
                values.append(-aeroblade(
                    img1=image, img2=rec["rec_B3HW"], net_type="vgg", reduction="none",
                    normalize=wrapper.range_after_transform == (0, 1)).cpu().reshape(-1))
        features[key] = torch.cat(values)
        codec.save_features(features, cache)
    print(f"  {CODEC} calibration {model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


def pack_calibration(model):
    feat = torch.load(path(model), map_location="cpu", weights_only=False)
    diff = (feat["cond_gt_llh_BL"] - feat["uncond_gt_llh_BL"]).float().numpy()
    lengths = feat.get("scale_lengths") or [diff.shape[1]]
    quant = feat["quant_err"].float().numpy()
    quant = np.stack([row.reshape(len(lengths), -1).mean(axis=1) for row in quant])
    return scale_tensor(diff, lengths, quant, feat["rec_err"].float().numpy())


def evaluate():
    original = {(ds, model): c2i.pack(ds, model)
                for ds in codec.DATASETS for model in codec.MODELS}
    compressed = {(ds, model): codec.pack_codec(ds, model)
                  for ds in codec.DATASETS for model in codec.MODELS}
    calibration = {model: pack_calibration(model) for model in codec.MODELS}
    ref_idx, select_idx, _, test_idx, train_idx, val_idx, fake_test = c2i.fixed_test_split(0)
    train_fake = {"var_d20": "var_d16", "var_d30": "var_d16",
                  "llamagen_b256": "llamagen_b256",
                  "llamagen_l256": "llamagen_b256", "rar_l": "rar_b"}
    combined = {ds: np.full(len(codec.codec_positions(ds)), -np.inf, dtype=np.float32)
                for ds in codec.DATASETS}
    combined_prada = dict(combined)
    cal_combined = np.full(len(CAL_IDX), -np.inf, dtype=np.float32)
    cal_prada = np.full(len(CAL_IDX), -np.inf, dtype=np.float32)
    for model in codec.MODELS:
        inputs = {"real": original["real", model], "fake": original[train_fake[model], model]}
        normalized = normalize_data(inputs, ref_idx)
        ref = np.concatenate([original["real", model][i] for i in ref_idx], axis=0)
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
        dart = train_dart(
            (np.concatenate([normalized["real"][ref_idx], normalized["fake"][train_idx]]),
             np.array([0] * c2i.N_REF_REAL + [1] * 100)),
            (np.concatenate([normalized["real"][select_idx], normalized["fake"][val_idx]]),
             np.array([0] * c2i.N_SELECT_REAL + [1] * 50)), 0)
        real_scores = predict(dart, normalized["real"])
        median = np.median(real_scores[select_idx])
        iqr = (np.percentile(real_scores[select_idx], 75)
               - np.percentile(real_scores[select_idx], 25) + 1e-6)
        normalized_cal = np.asarray([(x - mu) / sd for x in calibration[model]], dtype=object)
        cal_combined = np.maximum(cal_combined, (predict(dart, normalized_cal) - median) / iqr)
        cal_prada = np.maximum(cal_prada, prada_score_calibration(model))
        for ds in codec.DATASETS:
            normalized_test = np.asarray([(x - mu) / sd for x in compressed[ds, model]],
                                         dtype=object)
            combined[ds] = np.maximum(
                combined[ds], (predict(dart, normalized_test) - median) / iqr)
            combined_prada[ds] = np.maximum(
                combined_prada[ds], codec.codec_prada(ds, model))
    tau = conformal_tau(cal_combined)
    prada_tau = conformal_tau(cal_prada)
    reference = json.load(open(os.path.join(
        c2i.OUT, f"dart_c2i_{CODEC}_strict_fixed_test.json")))
    results = {"config": {"codec": CODEC, "seed": 0,
                          "protocol": "original train/selection; compressed calibration/test"},
               "generators": {}}
    max_difference = 0.0
    real_rows = codec.position_rows("real", test_idx)
    for gen in codec.FAKE_GENS:
        fake_rows = codec.position_rows(gen, fake_test)
        y = np.array([0] * c2i.N_TEST_REAL + [1] * 100)
        dart = np.concatenate([combined["real"][real_rows], combined[gen][fake_rows]])
        prada = np.concatenate([combined_prada["real"][real_rows],
                                combined_prada[gen][fake_rows]])
        results["generators"][gen] = {
            "dart": c2i.metric(y, dart, c2i.N_TEST_REAL, tau),
            "prada": c2i.metric(y, prada, c2i.N_TEST_REAL, prada_tau)}
        for method, score in (("dart", dart), ("prada", prada)):
            expected = reference["seeds"]["0"][gen][CODEC][method]["AUROC"]
            max_difference = max(max_difference, abs(
                roc_auc_score(y, score) - expected))
        np.savez(os.path.join(c2i.OUT, f"scores_dart_{CODEC}_compressed_cal_{gen}.npz"),
                 dart=dart, prada=prada, n_real=c2i.N_TEST_REAL,
                 dart_cal=cal_combined, prada_cal=cal_prada,
                 dart_threshold=tau, prada_threshold=prada_tau)
    results["summary"] = {
        "macro_AUROC": float(np.mean([
            results["generators"][gen]["dart"]["AUROC"] for gen in codec.FAKE_GENS])),
        "FPR": float(results["generators"]["var_d16"]["dart"]["FPR"]),
        "AUROC_max_difference_from_test_shift": max_difference}
    with open(os.path.join(c2i.OUT, f"dart_c2i_{CODEC}_compressed_cal.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def prada_score_calibration(model):
    feat = torch.load(path(model), map_location="cpu", weights_only=False)
    return codec.prada_score(feat, model)


def main():
    os.makedirs(OUT, exist_ok=True)
    for model in codec.MODELS:
        extract(model)
    if not EXTRACT_ONLY:
        evaluate()


if __name__ == "__main__":
    main()
