import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.device import autocast_ctx, get_device
from utils.logger import get_logger

from .evaluator import evaluate


class Trainer:
    def __init__(self, model, cfg, work_dir):
        self.cfg = cfg
        self.work_dir = work_dir
        self.device = get_device(cfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        tcfg = cfg.train
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=tcfg.lr,
            weight_decay=tcfg.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tcfg.epochs
        )
        self.amp = tcfg.get("amp", True)
        self.logger = get_logger("trainer", os.path.join(work_dir, "train.log"))

    def fit(self, train_set, val_set=None):
        tcfg = self.cfg.train
        train_loader = DataLoader(
            train_set, batch_size=tcfg.batch_size, shuffle=True,
            num_workers=tcfg.get("num_workers", 8), pin_memory=True, drop_last=True,
        )
        best = 0.0
        self.logger.info(
            f"训练集 {len(train_set)} 张, 验证集 "
            f"{len(val_set) if val_set is not None else 0} 张, device={self.device}"
        )
        for epoch in range(1, tcfg.epochs + 1):
            loss = self._train_epoch(train_loader, epoch)
            self.scheduler.step()
            msg = (f"epoch {epoch}/{tcfg.epochs} loss={loss:.4f} "
                   f"lr={self.scheduler.get_last_lr()[0]:.2e}")
            if val_set is not None:
                val_loader = DataLoader(
                    val_set, batch_size=self.cfg.eval.get("batch_size", 128),
                    shuffle=False, num_workers=tcfg.get("num_workers", 8),
                    pin_memory=True,
                )
                metrics = evaluate(self.model, val_loader, self.device, self.amp)["overall"]
                msg += " " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                score = metrics.get("ap", metrics["acc"])
                if score > best:
                    best = score
                    self._save("best.pth")
            self.logger.info(msg)
        self._save("last.pth")
        self.logger.info(f"训练完成, best={best:.4f}, 权重保存在 {self.work_dir}")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True)
            labels = batch["label"].float().to(self.device, non_blocking=True)
            with autocast_ctx(self.device, self.amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total / max(n, 1)

    def _save(self, name):
        torch.save(
            {"model": self.model.state_dict(), "cfg": dict(self.cfg)},
            os.path.join(self.work_dir, name),
        )
