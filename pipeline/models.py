"""One adapter per UQ method, all with the same interface so train_with_noise() and
evaluate_conformal() can take any of them.

An adapter is a *spec*: construct it with its hyperparameters (e.g. ``MCDropout(p=0.5)``)
and pass it to ``train_with_noise``. The networks themselves are created inside
``build(seed)`` so that weight initialisation is controlled by the run seed.

Interface
---------
name                       short identifier used in checkpoint / result file names
build(seed)                (re)create self.nets with seeded initialisation
nets                       list[nn.Module] that get trained (5 for DeepEnsemble)
member_seeds(seed)         training seed for each net
post_fit(train_noaug_loader)  post-hoc step after training (Laplace / SNGP)
predict_probs(loader, seed)   -> (probs [N, 10], labels [N]) numpy arrays
state_dict() / load_state_dict(sd)
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE


def _seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hub_resnet20():
    return torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
                          pretrained=False, trust_repo=True, verbose=False)


@torch.no_grad()
def _softmax_probs(nets, loader):
    """Mean softmax over `nets` (a single net gives the plain softmax)."""
    for n in nets:
        n.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        P.append(torch.stack([F.softmax(n(x), dim=1) for n in nets]).mean(0).cpu().numpy())
        Y.append(np.asarray(y))
    return np.concatenate(P), np.concatenate(Y)


class UQModel:
    name = "uq"

    def __init__(self):
        self.nets = []

    def _make_net(self):
        raise NotImplementedError

    def member_seeds(self, seed):
        return [seed]

    def build(self, seed=0):
        self.nets = []
        for s in self.member_seeds(seed):
            _seed_all(s)
            self.nets.append(self._make_net().to(DEVICE))
        return self

    def post_fit(self, train_noaug_loader):
        pass

    def predict_probs(self, loader, seed=0):
        return _softmax_probs(self.nets, loader)

    def config(self):
        return {}

    def state_dict(self):
        return {"nets": [n.state_dict() for n in self.nets]}

    def load_state_dict(self, sd):
        assert len(sd["nets"]) == len(self.nets), "checkpoint / model member count mismatch"
        for n, s in zip(self.nets, sd["nets"]):
            n.load_state_dict(s)

    def __repr__(self):
        cfg = ", ".join(f"{k}={v}" for k, v in self.config().items())
        return f"{type(self).__name__}({cfg})"


# ------------------------------- Baseline ------------------------------------
class Baseline(UQModel):
    """Plain cifar10_resnet20, single softmax pass."""
    name = "baseline"

    def _make_net(self):
        return _hub_resnet20()


# ------------------------------ MC Dropout -----------------------------------
# Ported from models/mc-dropout-model.ipynb (cell 7).
class MCDropoutResNet20(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        base = _hub_resnet20()
        self.conv1, self.bn1, self.relu = base.conv1, base.bn1, base.relu
        self.layer1, self.layer2, self.layer3 = base.layer1, base.layer2, base.layer3
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = base.fc

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(self.dropout(x))


def enable_mc_dropout(model):
    """Only Dropout modules in train mode; BatchNorm keeps running stats."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class MCDropout(UQModel):
    """Dropout before fc (Gal & Ghahramani 2016); prediction = mean of T stochastic passes."""
    name = "mc_dropout"

    def __init__(self, p=0.5, T=50):
        super().__init__()
        self.p, self.T = p, T

    def config(self):
        return {"p": self.p, "T": self.T}

    def _make_net(self):
        return MCDropoutResNet20(self.p)

    @torch.no_grad()
    def predict_probs(self, loader, seed=0):
        net = self.nets[0]
        net.eval()
        enable_mc_dropout(net)
        torch.manual_seed(seed)
        P, Y = [], []
        for x, y in loader:
            x = x.to(DEVICE)
            P.append(torch.stack([F.softmax(net(x), dim=1) for _ in range(self.T)])
                     .mean(0).cpu().numpy())
            Y.append(np.asarray(y))
        net.eval()
        return np.concatenate(P), np.concatenate(Y)


# ------------------------- Last-layer Laplace --------------------------------
# Ported from models/laplace-redux-model.ipynb (cells 11 and 13). Needs `laplace-torch`.
class LaplaceLL(UQModel):
    """MAP ResNet-20 + post-hoc last-layer Laplace (laplace-torch), GLM + probit predictive."""
    name = "laplace"

    def __init__(self, hessian_structure="full", prior_method="marglik",
                 pred_type="glm", link_approx="probit"):
        super().__init__()
        self.hessian_structure = hessian_structure
        self.prior_method = prior_method
        self.pred_type, self.link_approx = pred_type, link_approx
        self.la = None

    def config(self):
        return {"hessian_structure": self.hessian_structure, "prior_method": self.prior_method,
                "pred_type": self.pred_type, "link_approx": self.link_approx}

    def _make_net(self):
        return _hub_resnet20()

    def post_fit(self, train_noaug_loader):
        try:
            from laplace import Laplace
        except ImportError as e:
            raise ImportError("LaplaceLL needs laplace-torch: pip install laplace-torch==0.3") from e
        # laplace-torch has a CPU/GPU device mismatch building the Hessian; the last layer is
        # tiny, so fit on a CPU copy (same as the Laplace notebook).
        la_model = copy.deepcopy(self.nets[0]).cpu().eval()
        self.la = Laplace(la_model, "classification", subset_of_weights="last_layer",
                          hessian_structure=self.hessian_structure)
        self.la.fit(train_noaug_loader)
        self.la.optimize_prior_precision(method=self.prior_method)

    def predict_probs(self, loader, seed=0):
        if self.la is None:
            raise RuntimeError("call post_fit() (train_with_noise does this) before predicting")
        P, Y = [], []
        for x, y in loader:
            p = self.la(x, pred_type=self.pred_type, link_approx=self.link_approx)
            P.append(p.detach().cpu().numpy())
            Y.append(np.asarray(y))
        return np.concatenate(P), np.concatenate(Y)

    def map_probs(self, loader):
        """Plain softmax of the MAP network, for reference."""
        return _softmax_probs(self.nets, loader)


# ---------------------------------- SNGP -------------------------------------
# Ported from models/sngp-model.ipynb (cells 4, 5, 7 and 11).
class SpectralNormConv2d(nn.Conv2d):
    def __init__(self, *args, norm_bound=6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_bound = norm_bound
        self.register_buffer("u", F.normalize(torch.randn(self.weight.shape[0]), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(self.weight[0].numel()), dim=0))

    def forward(self, x):
        W = self.weight
        W_mat = W.reshape(W.shape[0], -1)
        if self.training:
            with torch.no_grad():
                v = F.normalize(W_mat.t() @ self.u, dim=0)
                u = F.normalize(W_mat @ v, dim=0)
                self.u.copy_(u)
                self.v.copy_(v)
        sigma = self.u @ (W_mat @ self.v)
        W = W * torch.clamp(self.norm_bound / sigma, max=1.0)
        return self._conv_forward(x, W, self.bias)


def to_sn(conv, c):
    new = SpectralNormConv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             bias=conv.bias is not None, norm_bound=c)
    new.weight.data.copy_(conv.weight.data)
    return new


def add_spectral_norm(model, c=6.0):
    for stage in (model.layer1, model.layer2, model.layer3):
        for block in stage:
            block.conv1 = to_sn(block.conv1, c)
            block.conv2 = to_sn(block.conv2, c)
    return model


class RFFGaussianProcess(nn.Module):
    """Random-feature GP output layer (SNGP, Liu et al. 2020, Sec. 3.1)."""

    def __init__(self, in_features, num_classes, num_features=1024,
                 length_scale=2.0, ridge=1.0, use_input_norm=True):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.length_scale = length_scale
        self.ridge = ridge
        self.feature_scale = math.sqrt(2.0 / num_features)
        self.input_norm = nn.LayerNorm(in_features) if use_input_norm else None
        self.register_buffer("W_rff", torch.randn(num_features, in_features))
        self.register_buffer("b_rff", 2 * math.pi * torch.rand(num_features))
        self.beta = nn.Linear(num_features, num_classes, bias=False)
        K, D = num_classes, num_features
        self.register_buffer("precision", torch.zeros(K, D, D))
        self.register_buffer("covariance", torch.eye(D).div(ridge).repeat(K, 1, 1))

    def features(self, h):
        if self.input_norm is not None:
            h = self.input_norm(h)
        h = h / self.length_scale
        return self.feature_scale * torch.cos(F.linear(h, self.W_rff, self.b_rff))

    def forward(self, h, return_var=False):
        phi = self.features(h)
        logits = self.beta(phi)
        if return_var:
            return logits, self.variance(phi)
        return logits

    def variance(self, phi):
        return torch.einsum("bd,kde,be->bk", phi, self.covariance, phi)

    @torch.no_grad()
    def reset_precision(self):
        self.precision.zero_()

    @torch.no_grad()
    def update_precision(self, h):
        phi = self.features(h)
        p = F.softmax(self.beta(phi), dim=-1)
        w = p * (1 - p)
        weighted = phi.unsqueeze(1) * w.unsqueeze(2)
        self.precision.add_(torch.einsum("bkd,be->kde", weighted, phi))

    @torch.no_grad()
    def finalize_covariance(self):
        eye = torch.eye(self.num_features, device=self.precision.device, dtype=torch.float64)
        prec = self.precision.double() + self.ridge * eye
        self.covariance.copy_(torch.linalg.inv(prec).to(self.covariance.dtype))

    @torch.no_grad()
    def predict_proba(self, h, num_samples=10):
        phi = self.features(h)
        mean = self.beta(phi)
        std = self.variance(phi).clamp_min(0).sqrt()
        eps = torch.randn(num_samples, *mean.shape, device=mean.device)
        return F.softmax(mean + std * eps, dim=-1).mean(dim=0)


def get_features(m, x):
    """Hub ResNet-20 forward minus fc."""
    x = m.relu(m.bn1(m.conv1(x)))
    x = m.layer3(m.layer2(m.layer1(x)))
    return m.avgpool(x).flatten(1)


def build_sngp_model(norm_bound=6.0, gp_features=1024, length_scale=2.0, ridge=1.0):
    model = add_spectral_norm(_hub_resnet20(), c=norm_bound)
    model.fc = RFFGaussianProcess(64, 10, num_features=gp_features,
                                  length_scale=length_scale, ridge=ridge)
    return model


class SNGP(UQModel):
    """Spectral-normalised ResNet-20 + random-feature GP head; covariance fitted post-hoc."""
    name = "sngp"

    def __init__(self, norm_bound=6.0, gp_features=1024, length_scale=2.0, ridge=1.0,
                 num_samples=10):
        super().__init__()
        self.norm_bound, self.gp_features = norm_bound, gp_features
        self.length_scale, self.ridge, self.num_samples = length_scale, ridge, num_samples

    def config(self):
        return {"norm_bound": self.norm_bound, "gp_features": self.gp_features,
                "length_scale": self.length_scale, "ridge": self.ridge,
                "num_samples": self.num_samples}

    def _make_net(self):
        return build_sngp_model(self.norm_bound, self.gp_features, self.length_scale, self.ridge)

    @torch.no_grad()
    def post_fit(self, train_noaug_loader):
        m = self.nets[0]
        m.eval()
        m.fc.reset_precision()
        for x, _ in train_noaug_loader:
            m.fc.update_precision(get_features(m, x.to(DEVICE)))
        m.fc.finalize_covariance()

    @torch.no_grad()
    def predict_probs(self, loader, seed=0):
        m = self.nets[0]
        m.eval()
        torch.manual_seed(seed)
        P, Y = [], []
        for x, y in loader:
            h = get_features(m, x.to(DEVICE))
            P.append(m.fc.predict_proba(h, self.num_samples).cpu().numpy())
            Y.append(np.asarray(y))
        return np.concatenate(P), np.concatenate(Y)


# ------------------------------ Deep Ensemble --------------------------------
class DeepEnsemble(UQModel):
    """n independently-trained ResNet-20s; prediction = mean of member softmaxes."""
    name = "deep_ensemble"

    def __init__(self, n=5):
        super().__init__()
        self.n = n

    def config(self):
        return {"n": self.n}

    def member_seeds(self, seed):
        return [seed * 100 + i for i in range(self.n)]

    def _make_net(self):
        return _hub_resnet20()


MODEL_REGISTRY = {c.name: c for c in (Baseline, MCDropout, LaplaceLL, SNGP, DeepEnsemble)}
