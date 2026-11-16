import torch.nn as nn


class LinearClassifier(nn.Module):
    """骨干网络 + 线性二分类头，输出单个 logit（配合 BCEWithLogitsLoss）。"""

    def __init__(self, backbone, feat_dim, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.dropout(feat)).squeeze(-1)
