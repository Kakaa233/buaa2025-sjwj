import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = 'resnet18', layer: str = 'avgpool'):
        super().__init__()
        if backbone == 'resnet18':
            try:
                net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                net = models.resnet18(weights=None)
        else:
            raise ValueError('Unsupported backbone')
        self.backbone = nn.Sequential(*(list(net.children())[:-1]))  # remove fc, keep to avgpool
        self.out_dim = net.fc.in_features
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)  # [B, C, 1, 1]
        return feat.flatten(1)   # [B, C]


class GaussianModel(nn.Module):
    """Simple class-conditional Gaussian density for features.
       Assumes one Gaussian for good and one for bad.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('mu_good', torch.zeros(dim))
        self.register_buffer('cov_good', torch.eye(dim))
        self.register_buffer('mu_bad', torch.zeros(dim))
        self.register_buffer('cov_bad', torch.eye(dim))
        self.dim = dim

    @staticmethod
    def _cov_regularize(cov, eps=1e-5):
        eye = torch.eye(cov.size(0), device=cov.device)
        return cov + eps * eye

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # X: [N, D], y: 0 good, 1 bad
        with torch.no_grad():
            Xg = X[y == 0]
            Xb = X[y == 1]
            mu_g = Xg.mean(0)
            mu_b = Xb.mean(0)
            Xg_c = Xg - mu_g
            Xb_c = Xb - mu_b
            cov_g = (Xg_c.T @ Xg_c) / max(1, Xg_c.size(0) - 1)
            cov_b = (Xb_c.T @ Xb_c) / max(1, Xb_c.size(0) - 1)
            self.mu_good.copy_(mu_g)
            self.mu_bad.copy_(mu_b)
            self.cov_good.copy_(self._cov_regularize(cov_g))
            self.cov_bad.copy_(self._cov_regularize(cov_b))

    def log_prob(self, X: torch.Tensor, good: bool = True):
        mu = self.mu_good if good else self.mu_bad
        cov = self.cov_good if good else self.cov_bad
        inv = torch.linalg.inv(cov)
        diff = X - mu
        maha = (diff @ inv * diff).sum(-1)
        sign, logdet = torch.slogdet(cov)
        const = -0.5 * (self.dim * torch.log(torch.tensor(2 * 3.1415926535, device=X.device)) + logdet)
        return const - 0.5 * maha

    def anomaly_score(self, X: torch.Tensor):
        # negative log-likelihood under 'good' minus under 'bad' -> higher means more anomalous
        lp_good = self.log_prob(X, good=True)
        lp_bad = self.log_prob(X, good=False)
        # score: higher -> more likely bad
        return (lp_bad - lp_good)
