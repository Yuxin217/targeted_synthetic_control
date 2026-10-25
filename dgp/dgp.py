"""Synthetic panel DGP for the paper's experiments.

One generator per outcome type, each with four outcome functions
(linear, hinge, quadratic, time_varying). Unit u000 is the treated unit,
assigned by design before any outcome is drawn.

    generate_panel(dgp, outcome, seed)  ->  long-format panel DataFrame
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

DGPS = ("linear", "hinge", "quadratic", "time_varying")
OUTCOMES = ("continuous", "binary")


@dataclass(frozen=True)
class PanelConfig:
    n_donors: int = 30
    n_covariates: int = 12
    last_pre_time: int = 40
    max_horizon: int = 10
    noise_sd: float = 0.8
    treatment_effect: float = 1.0
    # Continuous outcome
    signal_scale: float = 0.2
    # Binary outcome (logit scale)
    binary_intercept: float = -0.25
    binary_temperature: float = 2.0
    binary_spread: float = 0.35
    binary_latent: float = 0.2
    binary_drift: float = 0.9
    binary_rho: float = 0.2


# Binary linear DGP uses a wider covariate cohort and stronger latent factors.
BINARY_OVERRIDES = {"linear": {"binary_spread": 0.7, "binary_latent": 0.5}}


def _structural(dgp, x, u, drift, slope, nonlinearity, growth):
    """Structural mean g(x, t) for the four outcome functions."""
    g = (.8*x[:, 0] - .6*x[:, 1] + .35*x[:, 2])[:, None]
    g = g + (slope[0]*x[:, 3, None] - slope[1]*x[:, 4, None])*drift
    if dgp == "hinge":
        shape = .8*np.maximum(x[:, 0]-.25, 0) - .6*np.maximum(-x[:, 1]-.25, 0)
        g = g + nonlinearity*shape[:, None]*(1+growth*u)
    elif dgp == "quadratic":
        shape = .5*(x[:, 0]**2-1)/np.sqrt(2) + .4*x[:, 0]*x[:, 1]
        g = g + nonlinearity*shape[:, None]*(1+growth*u)
    elif dgp == "time_varying":
        wave = .7*np.sin(x[:, 0, None])*np.sin(2*np.pi*u)
        interaction = .5*x[:, 1, None]*x[:, 2, None]*np.cos(np.pi*u)
        # Kept in the archived evaluation order so panels are bit-identical.
        g = g + wave + interaction if nonlinearity == 1 else g + nonlinearity*(wave + interaction)
    return g


def _ar1(innovations, rho):
    ar = np.empty_like(innovations)
    ar[:, 0] = innovations[:, 0]
    for k in range(1, innovations.shape[1]):
        ar[:, k] = rho*ar[:, k-1] + np.sqrt(1-rho**2)*innovations[:, k]
    return ar


def _continuous(dgp, rng, cfg, n, periods, t, u):
    x = rng.standard_normal((n, cfg.n_covariates))
    innovations = rng.standard_normal((n, periods))
    g = _structural(dgp, x, u, u-.5, (.25, .15), 1., .2)
    signal = .2*u + .1*np.sin(2*np.pi*u) + cfg.signal_scale*g
    y0 = signal + cfg.noise_sd*innovations
    y = y0.copy()
    y[0] += cfg.treatment_effect*(t > cfg.last_pre_time)
    return x, y, y0, None


def _binary(dgp, rng, cfg, n, periods, t, u):
    opts = {"binary_spread": cfg.binary_spread, "binary_latent": cfg.binary_latent,
            **BINARY_OVERRIDES.get(dgp, {})}
    center = rng.standard_normal(cfg.n_covariates)
    x = center[None, :] + opts["binary_spread"]*rng.standard_normal((n, cfg.n_covariates))
    latent = rng.standard_normal((n, 4))
    innovations = rng.standard_normal((n, periods))
    uniforms = rng.random((n, periods))
    g = _structural(dgp, x, u, cfg.binary_drift*(u-.5), (.5, .3), 1.2, .3)
    factors = np.vstack([np.ones(periods), u-.5, np.sin(2*np.pi*u), np.cos(3*np.pi*u)])
    loadings = .35*x[:, 4:8] + opts["binary_latent"]*latent
    mean = .3*u + .2*np.sin(2*np.pi*u) + g + loadings@factors/2
    logits = cfg.binary_intercept + (mean + cfg.noise_sd*_ar1(innovations, cfg.binary_rho))/cfg.binary_temperature
    p0 = expit(logits)
    effect = cfg.treatment_effect*(1+.5*np.maximum(t-cfg.last_pre_time, 0)/cfg.max_horizon)
    p = p0.copy()
    p[0] = expit(logits[0] + effect*(t > cfg.last_pre_time))
    return x, (uniforms < p).astype(float), (uniforms < p0).astype(float), p0


def generate_panel(dgp, outcome, seed, config=PanelConfig()):
    """Long-format panel with columns x1..xp, abb, T, A, Y, Y0_true (+ Y0_prob_true)."""
    if dgp not in DGPS or outcome not in OUTCOMES:
        raise ValueError(f"dgp must be one of {DGPS}, outcome one of {OUTCOMES}")
    rng = np.random.default_rng(seed)
    n, periods = config.n_donors+1, config.last_pre_time+config.max_horizon
    t = np.arange(1, periods+1)
    u = t/config.last_pre_time
    make = _continuous if outcome == "continuous" else _binary
    x, y, y0, p0 = make(dgp, rng, config, n, periods, t, u)
    frame = pd.DataFrame(np.tile(x, (periods, 1)), columns=[f"x{k+1}" for k in range(config.n_covariates)])
    frame["abb"] = np.tile([f"u{k:03d}" for k in range(n)], periods)
    frame["T"] = np.repeat(t, n)
    frame["A"] = np.tile(np.arange(n) == 0, periods).astype(int)
    frame["Y"] = y.T.reshape(-1)
    frame["Y0_true"] = y0.T.reshape(-1)
    if p0 is not None:
        frame["Y0_prob_true"] = p0.T.reshape(-1)
    return frame
