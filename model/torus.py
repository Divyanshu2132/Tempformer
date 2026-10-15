import math

import numpy as np
import torch

TWO_PI = 2 * math.pi


def wrap(u):
    """To [-pi, pi)."""
    return torch.remainder(u + math.pi, TWO_PI) - math.pi


def score(d, sigma, n_images=10):
    """d/du log p(u_t | u_0) of the wrapped normal, with d = u_t - u_0 (any real) and sigma broadcastable to d."""
    k = torch.arange(-n_images, n_images + 1, device=d.device, dtype=d.dtype) * TWO_PI
    x = d.unsqueeze(-1) + k
    s = sigma.unsqueeze(-1) if torch.is_tensor(sigma) else sigma
    w = torch.softmax(-x ** 2 / (2 * s ** 2), dim=-1)
    return -(w * x).sum(-1) / (s.squeeze(-1) if torch.is_tensor(sigma) else s) ** 2


def _score_rms_table(sigma_min=1e-3, sigma_max=2 * math.pi, n_sigma=400, n_grid=4001):
    """RMS of the wrapped-normal score, E[s^2]^(1/2), on a log-spaced sigma grid (quadrature over the circle)."""
    sig = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), n_sigma))
    d = np.linspace(-np.pi, np.pi, n_grid)
    k = np.arange(-10, 11) * TWO_PI
    rms = []
    for s in sig:
        x = d[:, None] + k[None, :]
        logw = -x ** 2 / (2 * s ** 2)
        top = logw.max(1, keepdims=True)                       # log-space: no 0/0 far from the mode
        w = np.exp(logw - top)
        dens = np.exp(top[:, 0]) * w.sum(1)                    # unnormalised density on the circle
        sc = -(w * x).sum(1) / w.sum(1) / s ** 2
        rms.append(np.sqrt(np.trapz(sc ** 2 * dens, d) / np.trapz(dens, d)))
    return np.log(sig), np.array(rms)


_LOG_SIG, _RMS = _score_rms_table()


def score_rms(sigma):
    """RMS of the score at sigma (tensor), by interpolation in log sigma."""
    s = sigma.detach().cpu().double().log().numpy()
    return torch.as_tensor(np.interp(s, _LOG_SIG, _RMS), dtype=torch.float32, device=sigma.device)


def chi_period_factor(chi_pi_periodic):
    """(..., 4) 0/1 pi-periodicity -> factor 2 pi / period (1 or 2)."""
    return 1.0 + chi_pi_periodic
