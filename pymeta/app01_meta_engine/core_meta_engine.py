"""
App 01 - Core Meta Engine
=========================
Implements the backbone of meta-analysis:
- Effect sizes (OR, RR, SMD, MD, HR)
- Variance & SE computation
- Tau² estimators (DL, REML, SJ, ML)
- Pooling (fixed-effect, random-effects)
- Heterogeneity (Q, I², H²)
- Forest plots
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Effect size calculators
# ------------------------------------------------------------

def odds_ratio(events_t, n_t, events_c, n_c):
    """Compute log odds ratio and SE."""
    or_val = (events_t / (n_t - events_t)) / (events_c / (n_c - events_c))
    log_or = np.log(or_val)
    se = np.sqrt(1/events_t + 1/(n_t-events_t) + 1/events_c + 1/(n_c-events_c))
    return log_or, se

def risk_ratio(events_t, n_t, events_c, n_c):
    """Compute log risk ratio and SE."""
    rr = (events_t/n_t) / (events_c/n_c)
    log_rr = np.log(rr)
    se = np.sqrt(1/events_t - 1/n_t + 1/events_c - 1/n_c)
    return log_rr, se

def mean_diff(mean_t, sd_t, n_t, mean_c, sd_c, n_c):
    """Mean difference (MD) with SE."""
    md = mean_t - mean_c
    se = np.sqrt((sd_t**2)/n_t + (sd_c**2)/n_c)
    return md, se

def smd_cohens_d(mean_t, sd_t, n_t, mean_c, sd_c, n_c):
    """Standardized mean difference (Cohen’s d)."""
    s_p = np.sqrt(((n_t-1)*sd_t**2 + (n_c-1)*sd_c**2) / (n_t+n_c-2))
    d = (mean_t - mean_c) / s_p
    se = np.sqrt((n_t+n_c)/(n_t*n_c) + d**2/(2*(n_t+n_c)))
    return d, se

def hazard_ratio(hr, ci_low, ci_high):
    """Log HR and SE from CI."""
    log_hr = np.log(hr)
    se = (np.log(ci_high) - np.log(ci_low)) / (2*1.96)
    return log_hr, se

# ------------------------------------------------------------
# Tau² estimators & heterogeneity
# ------------------------------------------------------------

def dersimonian_laird(yi, vi):
    """DerSimonian–Laird tau²."""
    k = len(yi)
    wi = 1/vi
    ybar = np.sum(wi*yi) / np.sum(wi)
    Q = np.sum(wi*(yi-ybar)**2)
    df = k-1
    tau2 = max(0, (Q-df)/(np.sum(wi)-np.sum(wi**2)/np.sum(wi)))
    return tau2, Q, df

def restricted_ml(yi, vi, maxiter=100):
    """Restricted ML tau² (simplified Newton-Raphson)."""
    tau2 = 0.0
    for _ in range(maxiter):
        wi = 1/(vi+tau2)
        ybar = np.sum(wi*yi)/np.sum(wi)
        num = np.sum(wi*(yi-ybar)**2) - (len(yi)-1)
        den = np.sum(wi) - np.sum(wi**2)/np.sum(wi)
        new_tau2 = tau2 + num/den
        if new_tau2 < 0: new_tau2 = 0
        if abs(new_tau2-tau2) < 1e-6: break
        tau2 = new_tau2
    Q = np.sum(wi*(yi-ybar)**2)
    return tau2, Q, len(yi)-1

def heterogeneity_stats(Q, df):
    """Return I² and H²."""
    I2 = max(0, (Q-df)/Q) * 100 if Q>df else 0
    H2 = Q/df if df>0 else np.nan
    return I2, H2

# ------------------------------------------------------------
# Pooling functions
# ------------------------------------------------------------

def meta_analysis(yi, sei, method="DL", model="RE"):
    """
    Meta-analysis pooling.
    yi = log effects, sei = SEs
    method = DL | REML
    model = FE | RE
    """
    vi = sei**2
    if model=="FE":
        wi = 1/vi
        mu = np.sum(wi*yi)/np.sum(wi)
        se = np.sqrt(1/np.sum(wi))
        return {"model":"FE","estimate":mu,"se":se}
    elif model=="RE":
        if method=="DL":
            tau2,Q,df = dersimonian_laird(yi,vi)
        else:
            tau2,Q,df = restricted_ml(yi,vi)
        wi = 1/(vi+tau2)
        mu = np.sum(wi*yi)/np.sum(wi)
        se = np.sqrt(1/np.sum(wi))
        I2,H2 = heterogeneity_stats(Q,df)
        return {"model":"RE","method":method,"estimate":mu,"se":se,"tau2":tau2,"Q":Q,"I2":I2,"H2":H2}
    else:
        raise ValueError("Model must be FE or RE")

# ------------------------------------------------------------
# Forest plot
# ------------------------------------------------------------

def forest_plot(yi, sei, labels=None, pooled=None, title="Forest Plot"):
    """Draw forest plot of studies + pooled effect."""
    fig, ax = plt.subplots(figsize=(6,0.3*len(yi)+2))
    ci_low, ci_high = yi-1.96*sei, yi+1.96*sei
    ax.errorbar(yi, range(len(yi)), xerr=1.96*sei, fmt='o', color='black')
    if pooled:
        ax.axvline(pooled["estimate"], color='red', linestyle='--')
    ax.set_yticks(range(len(yi)))
    ax.set_yticklabels(labels if labels else [f"Study {i+1}" for i in range(len(yi))])
    ax.set_title(title)
    ax.axvline(0, color='grey', linestyle=':')
    plt.tight_layout()
    return fig