"""P0-A: RACE 复现与对齐评估 (v8 §9 P0-A, arXiv:2611.27161, 2026-11-20 预印本)。

复现级别声明 (重要): RACE 无官方代码, 本实现为**论文忠实近似版
(reimplementation-approximate)**, 以下点论文未完全指定, 采用文中描述的最
直接解释, 均在此记录:
  1. 门控/投影头/解码器的具体结构 (论文只给维度: shared 64, private 128)
     -> e_b=Linear(d,d) 跨深度共享; 门控 Linear(d,4) softmax;
        投影头 MLP(d->512->out); 重建解码器 MLP(192->512->768)
  2. 子损失定义 -> VICReg 风格: view=MSE, align=跨编码器 shared MSE,
     var=relu(1-std), cov=off-diag cov², sep=sh/pr 交叉协方差²,
     rec=重建 CLIP depth-24 锚点, geo=批量内成对余弦几何 MSE
  3. CLIP 深度特征: resblocks {6,12,18,24} 的裸 CLS token
     (中间深度无 ln_post/proj); DINOv2: get_intermediate_layers norm=True
  4. NP-RSR 描述子 (论文只给维度与口头描述):
     - conv6/11/16 taps: 2x2 avgpool 后去 16px 边界, 每通道
       [mean,std,min,max,median] -> 320/conv -> 960
     - 末端残差 24 维: 4 分布统计 + 4 方向自相关 + 4 相位象限能量
       + 12 径向谱能量
     - 掩码 AE: MLP 1968->1024->256->1024->1968, p=0.2
  5. 真实图像池: ARForensics train(6000)+val(1000) real (ImageNet 派生)
     替代论文的 mixed authentic dataset; 划分 train 3500 / dev 1000 /
     cal 2500 (论文校准用 5000, 本地真实池有限, 记录在案)
  6. 测试: ARForensics test 7 生成器 x (1000 real + 1000 fake) (seed 0,
     race-p0-1000示例), orig + JPEG95 两组

方法流程 (对齐论文 §3):
  CFE: DINOv2-L/14 + CLIP-L/14 冻结, 深度 {6,12,18,24}, 自适应多深度聚合
       (η=0.25 有界残差, γ 初始化 0), 共享/私有码, Eq.(2) 损失, 40 epoch
       AdamW 3e-4/1e-4; 检索 q=ν([√α_D d12; √α_C pr_C]) ∈ R^1152, K=16
  PRP: 3 个固定探针 (Gaussian/Uniform/Haar-HH, σ=1/255 RGB), DINOv2-L
       depth-12 余弦响应, dev median/IQR 归一
  NP-RSR: Noiseprint++ (TruFor 权重) 3 个原生 128x128 patch, 1968 维描述
       子, 掩码重建, 只对 24 个末端残差坐标计分
  融合: dev median/IQR 归一, s_G=(Z_a+s_P)/2, S=0.5 Z_sG + 0.5 Z_sR
  校准: cal real 上 conformal 阈值, 名义 FPR 5%

分阶段缓存: outputs/race/{feats,nprsr,cfe,aae}.npz|pt -> 断点续跑。

用法: python analysis/race_p0.py [--stage all]
"""

import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from scipy.stats import kurtosis, skew
from sklearn.metrics import average_precision_score, roc_auc_score, \
    roc_curve
from torchvision import transforms as T

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "TruFor",
                                "TruFor_train_test"))

from analysis.mpid_p0 import wait_for_gpu  # noqa: E402

AR_ROOT = os.path.join(PROJ, "data", "ARForensics")
OUT_DIR = os.path.join(PROJ, "outputs", "race")
NP_WEIGHTS = os.path.join(PROJ, "third_party", "TruFor", "TruFor_train_test",
                          "pretrained_models", "noiseprint++", "noiseprint++.th")
GENS = ["Infinity", "Janus_Pro", "LlamaGen", "Open_MAGVIT2", "RAR",
        "Switti", "VAR"]
DEPTHS = [6, 12, 18, 24]
DINO_IDX = [5, 11, 17, 23]
SEED = 0
N_TRAIN, N_DEV, N_CAL = 3500, 1000, 2500
N_TEST = 1000
K_NN = 16
ETA = 0.25

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_norm_dino = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_norm_clip = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
_to_tensor = T.ToTensor()


def jpeg95(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ================= S0: 数据清单 =================

def build_manifest():
    rng = np.random.RandomState(SEED)
    real_pool = []
    for split in ("train", "val"):
        d = os.path.join(AR_ROOT, split, "0_real")
        real_pool += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    perm = rng.permutation(len(real_pool))
    manifest = {
        "train_real": [real_pool[i] for i in perm[:N_TRAIN]],
        "dev_real": [real_pool[i]
                     for i in perm[N_TRAIN:N_TRAIN + N_DEV]],
        "cal_real": [real_pool[i]
                     for i in perm[N_TRAIN + N_DEV:
                                   N_TRAIN + N_DEV + N_CAL]],
        "test": {},
    }
    for g in GENS:
        gd = os.path.join(AR_ROOT, "test", g)
        for cls in ("0_real", "1_fake"):
            d = os.path.join(gd, cls)
            files = sorted(os.listdir(d))
            idx = np.random.RandomState(SEED).choice(
                len(files), min(N_TEST, len(files)), replace=False)
            manifest["test"][f"{g}/{cls}"] = [
                os.path.join(d, files[i]) for i in sorted(idx)]
    return manifest


# ================= S1: ViT 特征 + PRP =================

def load_backbones(device):
    import open_clip
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dino.eval().to(device)
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    clip = clip.visual.eval().to(device)
    for m in (dino, clip):
        for p in m.parameters():
            p.requires_grad = False
    return dino, clip


@torch.no_grad()
def dino_depth_cls(dino, x):
    inter = dino.get_intermediate_layers(
        x, n=DINO_IDX, norm=True, return_class_token=True)
    return torch.stack([c for _, c in inter], dim=1)  # (B,4,1024)


@torch.no_grad()
def clip_depth_cls(visual, x):
    """返回 (B,4,1024) 裸 CLS {6,12,18,24} 与最终 pooled (B,768)。
    注意: 当前 open_clip 版本全程 batch-first (NLD), 不要做 LND permute。"""
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    outs = []
    for i, blk in enumerate(visual.transformer.resblocks):
        x = blk(x)
        if i in DINO_IDX:
            outs.append(x[:, 0])
    x = visual.ln_post(x)
    pooled = x[:, 0]
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return torch.stack(outs, dim=1), pooled


def make_prp_probes(seed=7):
    rng = np.random.RandomState(seed)
    sigma = 1.0 / 255.0
    gau = rng.randn(224, 224, 3).astype(np.float32)
    gau *= sigma / gau.std()
    uni = rng.uniform(-1, 1, (224, 224, 3)).astype(np.float32)
    uni *= sigma / uni.std()
    checker = np.kron(np.ones((112, 112), dtype=np.float32),
                      np.array([[1.0, -1.0], [-1.0, 1.0]],
                               dtype=np.float32))[:, :, None]
    hh = (gau * checker).astype(np.float32)
    hh *= sigma / hh.std()
    return {"Gau": gau, "Uni": uni, "HH": hh}


@torch.no_grad()
def extract_vit(dino, clip, paths, device, batch_size=128, jpeg=False,
                prp_probes=None, tag=""):
    """v0 全局视图特征 (+ 可选 PRP 响应)。"""
    F_d, F_c, F_p, R = [], [], [], {t: [] for t in (prp_probes or {})}
    for bi in range(0, len(paths), batch_size):
        imgs01, imgs_j = [], []
        for p in paths[bi:bi + batch_size]:
            img = Image.open(p).convert("RGB").resize((224, 224),
                                                      Image.BICUBIC)
            if jpeg:
                img = jpeg95(img)
            imgs01.append(np.asarray(img, dtype=np.float32) / 255.0)
        x01 = torch.tensor(np.stack(imgs01)).permute(0, 3, 1, 2).float()
        xd = _norm_dino(x01.clone()).to(device)
        xc = _norm_clip(x01.clone()).to(device)
        F_d.append(dino_depth_cls(dino, xd).float().cpu())
        dc, pooled = clip_depth_cls(clip, xc)
        F_c.append(dc.float().cpu())
        F_p.append(pooled.float().cpu())
        if prp_probes:
            d0 = None
            for t, noise in prp_probes.items():
                xn = torch.clamp(x01 + torch.tensor(
                    noise).permute(2, 0, 1).unsqueeze(0).float(), 0, 1)
                xn = _norm_dino(xn).to(device)
                dt = Fn.normalize(
                    dino_depth_cls(dino, xn)[:, 1], dim=-1)
                if d0 is None:
                    d0 = Fn.normalize(
                        dino_depth_cls(dino, xd)[:, 1], dim=-1)
                r = 1 - (dt * d0).sum(-1)
                R[t].append(r.float().cpu())
        if bi == 0:
            print(f"    [{tag}] {len(paths)} 张提取中...", flush=True)
    out = {"dino": torch.cat(F_d).numpy(),
           "clip": torch.cat(F_c).numpy(),
           "clip_pooled": torch.cat(F_p).numpy()}
    for t in R:
        out[f"prp_{t}"] = torch.cat(R[t]).numpy()
    return out


@torch.no_grad()
def extract_vit_aug(dino, clip, paths, device, batch_size=128, tag=""):
    """CFE 训练第二视图: RRC(0.5-1.0)+hflip+轻微 jitter (固定 seed/图)。"""
    F_d, F_c = [], []
    for bi in range(0, len(paths), batch_size):
        xs_d, xs_c = [], []
        for p in paths[bi:bi + batch_size]:
            img = Image.open(p).convert("RGB")
            tf = T.Compose([
                T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2),
                T.ToTensor()])
            import hashlib
            torch.manual_seed(int(hashlib.md5(
                str(p).encode()).hexdigest()[:8], 16))
            x01 = tf(img)
            xs_d.append(_norm_dino(x01.clone()))
            xs_c.append(_norm_clip(x01.clone()))
        F_d.append(dino_depth_cls(dino, torch.stack(xs_d).to(device))
                   .float().cpu())
        dc, _ = clip_depth_cls(clip, torch.stack(xs_c).to(device))
        F_c.append(dc.float().cpu())
        if bi == 0:
            print(f"    [{tag}] {len(paths)} 张(aug)提取中...", flush=True)
    return {"dino": torch.cat(F_d).numpy(),
            "clip": torch.cat(F_c).numpy()}


# ================= S2: NP-RSR 描述子 =================

def build_noiseprint(device):
    from lib.models.DnCNN import make_net
    net = make_net(3, kernels=[3] * 17, features=[64] * 16 + [1],
                   bns=[False] + [True] * 15 + [False],
                   acts=["relu"] * 16 + ["linear"], dilats=[1] * 17,
                   bn_momentum=0.1, padding=1)
    dat = torch.load(NP_WEIGHTS, map_location="cpu")["network"]
    net.load_state_dict(dat)
    return net.eval().to(device)


def tap_features(net, x):
    """x (B,3,H,W) [0,1] -> conv6/11/16 后激活 taps 与末端残差。"""
    taps, ci, h = {}, 0, x
    for m in net:
        h = m(h)
        if isinstance(m, nn.Conv2d):
            ci += 1
            if ci == 17:
                taps[17] = h
        elif isinstance(m, nn.ReLU) and ci in (6, 11, 16):
            taps[ci] = h
    return taps[6], taps[11], taps[16], taps[17]


def channel_stats(t):
    """(B,C,H,W) -> 2x2 avgpool, 去16px边界(池化后8px), 每通道5统计。"""
    B, C, _, _ = t.shape
    p = Fn.avg_pool2d(t, 2)[:, :, 8:-8, 8:-8]
    p = p.reshape(B, C, -1)
    stats = [p.mean(-1), p.std(-1), p.min(-1).values, p.max(-1).values,
             p.median(-1).values]
    return torch.stack(stats, dim=-1).reshape(B, -1)  # (B, C*5)


def residual_stats(r):
    """(B,1,H,W) 末端残差 -> 24 维统计 (去16px边界)。"""
    B = r.shape[0]
    out = []
    for i in range(B):
        x = r[i, 0, 16:-16, 16:-16].cpu().numpy()
        flat = x.ravel()
        if flat.std() < 1e-12:
            s = [float(flat.mean()), 0.0, 0.0, 0.0]
        else:
            s = [flat.mean(), flat.std(), float(skew(flat)),
                 float(kurtosis(flat))]
        # 方向自相关 lag1 (零方差平坦区域 -> 0)
        for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = x[:max(0, x.shape[0] - abs(dx)),
                  :max(0, x.shape[1] - abs(dy))]
            b = np.roll(np.roll(x, dx, 0), dy, 1)[
                :max(0, x.shape[0] - abs(dx)),
                :max(0, x.shape[1] - abs(dy))]
            af, bf = a.ravel(), b.ravel()
            if af.std() < 1e-12 or bf.std() < 1e-12:
                s.append(0.0)
            else:
                s.append(float(np.corrcoef(af, bf)[0, 1]))
        # 相位象限能量
        F = np.fft.fftshift(np.fft.fft2(x))
        mag2 = np.abs(F) ** 2
        ph = np.angle(F)
        for q in range(4):
            mask = (ph >= -np.pi + q * np.pi / 2) & (ph < -np.pi + (q + 1) * np.pi / 2)
            s.append(float(np.log1p(mag2[mask].mean())))
        # 径向谱能量 (12 带)
        h, w = x.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = rad.max()
        total = mag2.sum() + 1e-12
        for b in range(12):
            mask = (rad >= b * rmax / 12) & (rad < (b + 1) * rmax / 12)
            s.append(float(mag2[mask].sum() / total))
        out.append(s[:24])
    return torch.tensor(np.array(out), dtype=torch.float32)


@torch.no_grad()
def nprsr_descriptors(net, paths, device, jpeg=False, batch_size=32,
                      tag=""):
    """每图 3 个原生 128x128 patch (center/top-left/bottom-right),
    返回 (N,3,1968)。"""
    probe = np.random.RandomState(11).randn(128, 128, 3).astype(np.float32)
    probe *= (1.0 / 255.0) / probe.std()
    outs = []
    for bi in range(0, len(paths), batch_size):
        vs = []
        for p in paths[bi:bi + batch_size]:
            img = Image.open(p).convert("RGB")
            if min(img.size) < 128:
                w, h = img.size
                sc = 256 / min(w, h)
                img = img.resize((int(w * sc), int(h * sc)), Image.BICUBIC)
            if jpeg:
                img = jpeg95(img)
            a = np.asarray(img, dtype=np.float32) / 255.0
            H, W = a.shape[:2]
            cy, cx = H // 2, W // 2
            coords = [(cy - 64, cx - 64), (0, 0), (H - 128, W - 128)]
            pv = []
            for (y, x0) in coords:
                patch = a[y:y + 128, x0:x0 + 128]
                psi = _psi_of(net, patch, device)
                pert = np.clip(patch + probe, 0, 1)
                psi_p = _psi_of(net, pert, device)
                pv.append(np.concatenate([psi, psi_p - psi]))
            vs.append(np.stack(pv))
        outs.append(np.stack(vs))
        if bi == 0:
            print(f"    [{tag}] {len(paths)} 张描述子提取中...", flush=True)
    return np.concatenate(outs)


def _psi_of(net, patch, device):
    x = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        t6, t11, t16, res = tap_features(net, x)
        ints = torch.cat([channel_stats(t) for t in (t6, t11, t16)], dim=1)
        res24 = residual_stats(res).to(device)
    psi = torch.cat([ints, res24], dim=1).squeeze(0).cpu().numpy()
    return psi


# ================= S3: CFE 模块 =================

class EncoderBranch(nn.Module):
    def __init__(self, dim, n_depth=4):
        super().__init__()
        self.e = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.gate_sh = nn.Linear(dim, n_depth)
        self.gate_pr = nn.Linear(dim, n_depth)
        self.gamma_sh = nn.Parameter(torch.zeros(1))
        self.gamma_pr = nn.Parameter(torch.zeros(1))
        self.head_sh = nn.Sequential(nn.Linear(dim, 512), nn.GELU(),
                                     nn.Linear(512, 64))
        self.head_pr = nn.Sequential(nn.Linear(dim, 512), nn.GELU(),
                                     nn.Linear(512, 128))
        self.register_buffer("mu_sh", torch.zeros(dim))
        self.register_buffer("sd_sh", torch.ones(dim))
        self.register_buffer("mu_pr", torch.zeros(dim))
        self.register_buffer("sd_pr", torch.ones(dim))

    def adapt(self, feats, route):
        g_in = self.ln(feats).mean(1)
        gate = self.gate_sh if route == "sh" else self.gate_pr
        gamma = self.gamma_sh if route == "sh" else self.gamma_pr
        anchor = feats[:, 3] if route == "sh" else feats[:, 1]
        pi = torch.softmax(gate(g_in), dim=-1).unsqueeze(-1)
        v = (pi * self.e(feats)).sum(1)
        return anchor + ETA * torch.tanh(gamma) * \
            anchor.norm(dim=-1, keepdim=True).detach() * Fn.normalize(v, dim=-1)

    def forward(self, feats):
        f_sh = self.adapt(feats, "sh")
        f_pr = self.adapt(feats, "pr")
        z_sh = (f_sh - self.mu_sh) / self.sd_sh
        z_pr = (f_pr - self.mu_pr) / self.sd_pr
        return self.head_sh(z_sh), self.head_pr(z_pr), f_sh, f_pr


def _var_loss(z, target=1.0):
    return Fn.relu(target - z.std(dim=0)).mean()


def _cov_offdiag(z):
    z = z - z.mean(0)
    c = (z.T @ z) / (len(z) - 1)
    off = c - torch.diag(torch.diag(c))
    return (off ** 2).sum() / z.shape[1]


def _geo_loss(z, ref):
    zn = Fn.normalize(z, dim=-1)
    rn = Fn.normalize(ref, dim=-1)
    return Fn.mse_loss(zn @ zn.T, rn @ rn.T)


def train_cfe(feats_tr_v0, feats_tr_v1, device, epochs=40, batch=256,
              lr=3e-4, wd=1e-4):
    br = {b: EncoderBranch(feats_tr_v0[b].shape[-1]).to(device)
          for b in ("dino", "clip")}
    # 用 v0 锚点统计做坐标标准化 (gamma=0 时 adapted=anchor)
    for b in ("dino", "clip"):
        f = torch.tensor(feats_tr_v0[b])
        br[b].mu_sh.copy_(f[:, 3].mean(0))
        br[b].sd_sh.copy_(f[:, 3].std(0) + 1e-6)
        br[b].mu_pr.copy_(f[:, 1].mean(0))
        br[b].sd_pr.copy_(f[:, 1].std(0) + 1e-6)
    dec = nn.Sequential(nn.Linear(192, 512), nn.GELU(),
                        nn.Linear(512, feats_tr_v0["clip"].shape[-1])).to(device)
    params = list(dec.parameters())
    for b in br.values():
        params += list(b.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    X = {b: torch.tensor(feats_tr_v0[b], device=device) for b in br}
    Y = {b: torch.tensor(feats_tr_v1[b], device=device) for b in br}
    n = len(X["dino"])
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = {}
            anchors = {}
            for b in br:
                sh0, pr0, f_sh0, _ = br[b](X[b][idx])
                sh1, pr1, _, _ = br[b](Y[b][idx])
                out[b] = (sh0, pr0, sh1, pr1, f_sh0)
                anchors[b] = X[b][idx][:, 3]
            shC, prC, shC1, prC1, f_shC = out["clip"]
            shD, prD, shD1, prD1, f_shD = out["dino"]
            l_view_sh = sum(Fn.mse_loss(out[b][0], out[b][2]) for b in br) / 2
            l_view_pr = sum(Fn.mse_loss(out[b][1], out[b][3]) for b in br) / 2
            l_align = (Fn.mse_loss(shC, shD) + Fn.mse_loss(shC1, shD1)) / 2
            l_varC = (_var_loss(shC) + _var_loss(prC)) / 2
            l_covC = _cov_offdiag(torch.cat([shC, prC], dim=1))
            l_sep = sum(_cov_offdiag(torch.cat([
                Fn.normalize(out[b][0], dim=-1),
                Fn.normalize(out[b][1], dim=-1)], dim=1)) for b in br) / 2
            l_recC = Fn.mse_loss(dec(torch.cat([shC, prC], dim=1)),
                                 anchors["clip"].detach())
            l_geoC = _geo_loss(shC, anchors["clip"].detach())
            loss = (10 * l_view_sh + 2.5 * l_view_pr + 10 * l_align
                    + 25 * l_varC + l_covC + l_sep
                    + 5 * l_recC + l_geoC)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [cfe] epoch {ep}: loss={tot / n:.4f}", flush=True)
    return br, dec


# ================= S4: 分数 =================

def iqr_scale(x, floor=1e-3):
    q1, q3 = np.percentile(x, [25, 75])
    return max((q3 - q1) / 1.349, floor)


def knn_mean_cos(bank, q, k=K_NN, chunk=2048):
    """q, bank 均已单位归一; 返回每个 q 的 kNN 平均余弦距离。"""
    sims = []
    for i in range(0, len(q), chunk):
        s = q[i:i + chunk] @ bank.T
        top = np.partition(-s, k, axis=1)[:, :k]
        sims.append(-top.mean(1))
    return np.concatenate(sims)


def zscore_med(x, med, scale):
    return (x - med) / scale


# ================= S5: masked AE =================

class DescAE(nn.Module):
    def __init__(self, dim=1968, latent=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(),
                                 nn.Linear(1024, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 1024), nn.GELU(),
                                 nn.Linear(1024, dim))

    def forward(self, z):
        h = self.enc(z)
        return self.dec(h), h


def train_desc_ae(Z, device, epochs=40, batch=256, lr=3e-4, wd=1e-4,
                  mask_p=0.2):
    ae = DescAE().to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=wd)
    Zt = torch.tensor(Z, device=device)
    n = len(Zt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            zb = Zt[perm[i:i + batch]]
            mask = (torch.rand_like(zb) > mask_p).float() / (1 - mask_p)
            zh, h = ae(zb * mask)
            l_rec = Fn.mse_loss(zh, zb)
            l_geo = _geo_loss(h, zb)
            l_var = Fn.relu(0.5 - h.std(dim=0)).mean()
            l_cov = _cov_offdiag(h)
            loss = l_rec + 0.1 * l_geo + 0.1 * l_var + 0.001 * l_cov
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(zb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [nprsr-ae] epoch {ep}: loss={tot / n:.4f}", flush=True)
    return ae.eval()


# ================= 主流程 =================

def save_npz(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in d.items()})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man_path = os.path.join(OUT_DIR, "manifest.json")
    if os.path.exists(man_path):
        manifest = json.load(open(man_path))
    else:
        manifest = build_manifest()
        json.dump(manifest, open(man_path, "w"))
    print("S0 数据清单就绪", flush=True)

    # ---- S1/S2: 特征提取 ----
    feats_path = os.path.join(OUT_DIR, "feats.npz")
    nprsr_path = os.path.join(OUT_DIR, "nprsr.npz")
    if not (os.path.exists(feats_path) and os.path.exists(nprsr_path)):
        gpu = wait_for_gpu(need_mb=5500)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = None

    if not os.path.exists(feats_path):
        print("S1 提取 ViT 特征 (DINOv2-L/14 + CLIP-L/14)...", flush=True)
        dino, clip = load_backbones(device)
        probes = make_prp_probes()
        feats = {}
        sets = {"train_real": manifest["train_real"],
                "dev_real": manifest["dev_real"],
                "cal_real": manifest["cal_real"]}
        for key, paths in sets.items():
            print(f"  {key} ({len(paths)})", flush=True)
            feats[f"{key}::v0"] = extract_vit(
                dino, clip, paths, device, prp_probes=probes, tag=key)
            feats[f"{key}::v1"] = extract_vit_aug(dino, clip, paths,
                                                  device, tag=key)
        for g in GENS:
            for cls in ("0_real", "1_fake"):
                key = f"{g}/{cls}"
                for codec in ("orig", "jpeg95"):
                    print(f"  {key} {codec}", flush=True)
                    feats[f"{key}::{codec}"] = extract_vit(
                        dino, clip, manifest["test"][key], device,
                        jpeg=(codec == "jpeg95"), prp_probes=probes,
                        tag=f"{key}/{codec}")
        save_npz(feats_path, {f"{k}::{kk}": vv for k, v in feats.items()
                              for kk, vv in v.items()})
        del dino, clip
        torch.cuda.empty_cache()
        print("S1 完成", flush=True)

    if not os.path.exists(nprsr_path):
        print("S2 提取 NP-RSR 描述子...", flush=True)
        net = build_noiseprint(device)
        nprsr = {}
        sets = {"train_real": manifest["train_real"],
                "dev_real": manifest["dev_real"],
                "cal_real": manifest["cal_real"]}
        for key, paths in sets.items():
            print(f"  {key}", flush=True)
            nprsr[key] = nprsr_descriptors(net, paths, device, tag=key)
        for g in GENS:
            for cls in ("0_real", "1_fake"):
                key = f"{g}/{cls}"
                for codec in ("orig", "jpeg95"):
                    print(f"  {key} {codec}", flush=True)
                    nprsr[f"{key}::{codec}"] = nprsr_descriptors(
                        net, manifest["test"][key], device,
                        jpeg=(codec == "jpeg95"), tag=f"{key}/{codec}")
        save_npz(nprsr_path, nprsr)
        del net
        torch.cuda.empty_cache()
        print("S2 完成", flush=True)

    feats = dict(np.load(feats_path))
    nprsr = dict(np.load(nprsr_path))
    n_nan = sum(int(np.isnan(v).sum()) for v in nprsr.values())
    if n_nan:
        print(f"NP-RSR 描述子含 {n_nan} 个 NaN (平坦 patch 的零方差相关), "
              f"nan_to_num 清洗", flush=True)
        nprsr = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                 for k, v in nprsr.items()}

    def get(prefix, codec="orig"):
        return {"dino": feats[f"{prefix}::{codec}::dino"],
                "clip": feats[f"{prefix}::{codec}::clip"],
                "clip_pooled": feats[f"{prefix}::{codec}::clip_pooled"],
                "prp": {t: feats[f"{prefix}::{codec}::prp_{t}"]
                        for t in ("Gau", "Uni", "HH")}}

    # ---- S3: CFE 训练 (CPU/GPU 皆可, 特征已缓存) ----
    cfe_path = os.path.join(OUT_DIR, "cfe.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fv0 = {"dino": feats["train_real::v0::dino"],
           "clip": feats["train_real::v0::clip"]}
    fv1 = {"dino": feats["train_real::v1::dino"],
           "clip": feats["train_real::v1::clip"]}
    if os.path.exists(cfe_path):
        ck = torch.load(cfe_path, map_location=device, weights_only=False)
        br = ck["br"]
        print("S3 加载缓存 CFE", flush=True)
    else:
        print("S3 训练 CFE (real-only)...", flush=True)
        br, dec = train_cfe(fv0, fv1, device)
        torch.save({"br": br}, cfe_path)

    @torch.no_grad()
    def codes(feats_one):
        out = {}
        for b in ("dino", "clip"):
            x = torch.tensor(feats_one[b], device=device)
            sh, pr, _, _ = br[b](x)
            out[b] = (sh.cpu().numpy(), pr.cpu().numpy())
        return out

    # ---- S4: 检索分数 a(x) / PRP s_P / NP-RSR s_R ----
    print("S4 计算分数...", flush=True)
    # dev 上的分量尺度 -> α 与 Z 统计
    dev = get("dev_real", "v0")
    dev_codes = codes(dev)
    d12_dev = Fn.normalize(torch.tensor(dev["dino"][:, 1]),
                           dim=-1).numpy()
    c_dev = Fn.normalize(torch.tensor(dev_codes["clip"][1]),
                         dim=-1).numpy()
    bank_dev_d = d12_dev  # dev 自检索估计 ω (排除自身 k+1)
    omega_d = iqr_scale(knn_mean_cos(bank_dev_d, d12_dev, k=K_NN + 1))
    omega_c = iqr_scale(knn_mean_cos(c_dev, c_dev, k=K_NN + 1))
    alpha_d = (1 / omega_d) / (1 / omega_d + 1 / omega_c)
    alpha_c = 1 - alpha_d
    print(f"  alpha_d={alpha_d:.3f} alpha_c={alpha_c:.3f}", flush=True)

    def qvec(feats_one):
        cs = codes(feats_one)
        d12 = Fn.normalize(torch.tensor(feats_one["dino"][:, 1]),
                           dim=-1).numpy()
        c = Fn.normalize(torch.tensor(cs["clip"][1]), dim=-1).numpy()
        q = np.concatenate([np.sqrt(alpha_d) * d12,
                            np.sqrt(alpha_c) * c], axis=1)
        return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(
            np.float32)

    bank = qvec(fv0)

    # PRP Z 统计 (dev)
    prp_stats = {t: (float(np.median(dev["prp"][t])),
                     iqr_scale(dev["prp"][t])) for t in dev["prp"]}

    def prp_score(feats_one):
        zs = [zscore_med(feats_one["prp"][t], *prp_stats[t])
              for t in ("Gau", "Uni", "HH")]
        return (zs[0] + zs[1] + zs[2]) / 3

    # NP-RSR: 标准化 + 训练 masked AE
    tr_desc = nprsr["train_real"].reshape(-1, 1968)
    mu_d, sd_d = tr_desc.mean(0), np.maximum(tr_desc.std(0), 1e-3)

    def zclip(v):
        return np.clip((v - mu_d) / sd_d, -15, 15)

    ae_path = os.path.join(OUT_DIR, "nprsr_ae.pt")
    if os.path.exists(ae_path):
        ae = DescAE().to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device,
                                      weights_only=True))
        ae.eval()
        print("  加载缓存 NP-RSR AE", flush=True)
    else:
        print("  训练 NP-RSR masked AE (real-only)...", flush=True)
        ae = train_desc_ae(zclip(tr_desc), device)
        torch.save(ae.state_dict(), ae_path)

    RES_IDX = np.arange(960, 984)  # 末端残差 24 坐标

    @torch.no_grad()
    def rsr_score(desc):
        z = torch.tensor(zclip(desc.reshape(-1, 1968)), device=device)
        zh, _ = ae(z)
        err = ((zh - z)[:, RES_IDX] ** 2).mean(1).cpu().numpy()
        return err.reshape(-1, 3).mean(1)

    def all_scores(feats_one, desc):
        q = qvec(feats_one)
        a = knn_mean_cos(bank, q, k=K_NN)
        return {"a": a, "sP": prp_score(feats_one), "sR": rsr_score(desc)}

    dev_s = all_scores(dev, nprsr["dev_real"])
    z_a = (float(np.median(dev_s["a"])), iqr_scale(dev_s["a"]))
    sG_dev = (zscore_med(dev_s["a"], *z_a) + dev_s["sP"]) / 2
    z_sG = (float(np.median(sG_dev)), iqr_scale(sG_dev))
    z_sR = (float(np.median(dev_s["sR"])), iqr_scale(dev_s["sR"]))

    def final_score(s):
        sG = (zscore_med(s["a"], *z_a) + s["sP"]) / 2
        return 0.5 * zscore_med(sG, *z_sG) + 0.5 * zscore_med(
            s["sR"], *z_sR)

    # 校准 (conformal, 名义 FPR 5%)
    cal_s = all_scores(get("cal_real", "v0"), nprsr["cal_real"])
    S_cal = final_score(cal_s)
    n_cal = len(S_cal)
    j = int(np.ceil((n_cal + 1) * 0.95))
    tau = float(np.sort(S_cal)[min(j, n_cal) - 1])
    print(f"  校准阈值 tau={tau:.4f} (n={n_cal}, j={j})", flush=True)

    # ---- S5: 测试评估 ----
    print("S5 测试评估...", flush=True)
    results = {"config": {
        "reproduction": "reimplementation-approximate (无官方代码, 见脚本头)",
        "protocol": "race-p0-1000示例 (ARForensics test, seed 0)",
        "alpha": {"dino": float(alpha_d), "clip": float(alpha_c)},
        "tau_fpr5": tau, "n_cal": n_cal,
    }, "per_generator": {}}
    for g in GENS:
        entry = {}
        for codec in ("orig", "jpeg95"):
            fr = get(f"{g}/0_real", codec)
            ff = get(f"{g}/1_fake", codec)
            Sr = final_score(all_scores(fr, nprsr[f"{g}/0_real::{codec}"]))
            Sf = final_score(all_scores(ff, nprsr[f"{g}/1_fake::{codec}"]))
            y = np.array([0] * len(Sr) + [1] * len(Sf))
            S = np.concatenate([Sr, Sf])
            fpr, tpr, _ = roc_curve(y, S)
            tpr_at_1 = float(np.interp(0.01, fpr, tpr))
            entry[codec] = {
                "AUROC": float(roc_auc_score(y, S)),
                "AP": float(average_precision_score(y, S)),
                "TPR@1%FPR": tpr_at_1,
                "FPR@tau(real)": float((Sr > tau).mean()),
            }
            # 分量 AUC (诊断)
            for comp in ("a", "sP", "sR"):
                sr = all_scores(fr, nprsr[f"{g}/0_real::{codec}"])[comp] \
                    if comp != "sP" else prp_score(fr)
                sf = all_scores(ff, nprsr[f"{g}/1_fake::{codec}"])[comp] \
                    if comp != "sP" else prp_score(ff)
                entry[codec][f"AUROC_{comp}"] = float(roc_auc_score(
                    y, np.concatenate([sr, sf])))
        results["per_generator"][g] = entry
        print(f"  {g}: " + json.dumps(
            {c: {k: round(v, 4) for k, v in e.items()
                 if not k.startswith("AUROC_")}
             for c, e in entry.items()}), flush=True)

    for codec in ("orig", "jpeg95"):
        aucs = [results["per_generator"][g][codec]["AUROC"] for g in GENS]
        results.setdefault("macro", {})[codec] = {
            "AUROC": float(np.mean(aucs)), "worst": float(np.min(aucs))}
    print(json.dumps(results["macro"], indent=2), flush=True)

    out_path = os.path.join(OUT_DIR, "race_p0.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {out_path}", flush=True)


if __name__ == "__main__":
    main()
