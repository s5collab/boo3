"""
01b_gmm_on_geha26.py — apply the SAME 11-parameter GMM (priors, model, sampler
settings) used for our S5 analysis to Geha+26's Boo III catalog. Output goes to
output/boo3_gmm_geha26_samples.h5 + a short text summary.

This implements the cross-check that the paper text in Section 3.4 claims
("re-running our GMM on the Geha+26 catalog ... v_hel = 191.30, σ_v = 2.95").
"""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from astropy.io import fits
from astropy import table as atable
from astropy.coordinates import SkyCoord
import astropy.units as u
import emcee

warnings.filterwarnings("ignore", category=UserWarning)

NB_DIR = Path(__file__).resolve().parent
PROJ   = NB_DIR.parent
DATA_DIR = PROJ / "data"
OUT_DIR  = PROJ / "output"

# --- import the EXACT model from 01_compute_gmm.py ---
sys.path.insert(0, str(NB_DIR))
import importlib.util
spec = importlib.util.spec_from_file_location("compute_gmm", str(NB_DIR / "01_compute_gmm.py"))
m = importlib.util.module_from_spec(spec)
# Avoid running the main() block; only import constants and functions
spec.loader.exec_module(m)

PARAM_NAMES   = m.PARAM_NAMES
PRIOR_BOUNDS  = m.PRIOR_BOUNDS
PM_BOX        = m.PM_BOX
VHEL_BOX      = m.VHEL_BOX
FEH_BOX       = m.FEH_BOX
log_likelihood = m.log_likelihood
membership_prob = m.membership_prob

print("=" * 70)
print("01b_gmm_on_geha26.py — same GMM as section:input_sample, applied to Geha+26")
print("=" * 70)
print(f"PRIOR_BOUNDS = {PRIOR_BOUNDS}")
print(f"PM_BOX  = {PM_BOX}")
print(f"VHEL_BOX = {VHEL_BOX}, FEH_BOX = {FEH_BOX}")
print()

# --- Load Geha+26 data ---
t = atable.Table.read(DATA_DIR / "geha2026_keck_bootes_3.fits")
print(f"Geha+26 raw rows: {len(t)}")

# Quality + Pmem cut: use Pmem > 0.5 as the loose box analog (matches Geha+26's
# membership threshold). We then apply the same box cuts as section input_sample,
# minus the S5 S/N cut and the 3 r_h spatial cut (Geha+26 is a single-mask).
# We also exclude Gaia DR3 RRL cross-matches (RRL_GEHA), per the paper text.
RRL_GEHA = 1450796178282259072

import numpy as np
# Use ALL Geha+26 catalog stars (no Pmem cut here — we want the GMM to do the classification)
# but apply the loose box that defines our input catalog.
need = ['v','v_err','ew_feh','ew_feh_err',
        'gaia_pmra','gaia_pmra_err','gaia_pmdec','gaia_pmdec_err','gaia_pmra_pmdec_corr',
        'gaia_source_id']
mask = np.ones(len(t), bool)
for c in need:
    arr = np.asarray(t[c])
    mask &= np.isfinite(arr)
    if 'err' in c: mask &= (arr > 0)
mask &= (t['gaia_source_id'] != -999)  # require Gaia PM match (otherwise no PM data)
mask &= (t['gaia_source_id'] != RRL_GEHA)  # exclude RRL cross-match
mask &= (t['v'] > VHEL_BOX[0]) & (t['v'] < VHEL_BOX[1])
mask &= (t['ew_feh'] > FEH_BOX[0]) & (t['ew_feh'] < FEH_BOX[1])
mask &= (t['gaia_pmra'] > PM_BOX[0]) & (t['gaia_pmra'] < PM_BOX[1])
mask &= (t['gaia_pmdec'] > PM_BOX[2]) & (t['gaia_pmdec'] < PM_BOX[3])
# mask &= (t['v_err'] < 10)  # COMMENTED: too strict for Geha+26 BHB stars

t_in = t[mask]
print(f"After (Gaia PM available) + (no RRL) + (RV/FeH/PM boxes) + (v_err < 10): {len(t_in)} stars")
print(f"  v range: {min(t_in['v']):.1f} – {max(t_in['v']):.1f}")
print(f"  ew_feh range: {min(t_in['ew_feh']):.2f} – {max(t_in['ew_feh']):.2f}")

# Map to column names expected by the GMM
df_in = pd.DataFrame({
    'vel_calib':       np.asarray(t_in['v']),
    'vel_calib_std':   np.asarray(t_in['v_err']),
    'feh50':           np.asarray(t_in['ew_feh']),
    'feh_calib_std':   np.asarray(t_in['ew_feh_err']),
    'pmra':            np.asarray(t_in['gaia_pmra']),
    'pmdec':           np.asarray(t_in['gaia_pmdec']),
    'pmra_error':      np.asarray(t_in['gaia_pmra_err']),
    'pmdec_error':     np.asarray(t_in['gaia_pmdec_err']),
    'pmra_pmdec_corr': np.asarray(t_in['gaia_pmra_pmdec_corr']),
    'source_id':       np.asarray(t_in['gaia_source_id']).astype('int64'),
})
print(f"GMM input frame: {len(df_in)} stars")

# --- Run emcee with EXACT same settings as 01_compute_gmm.py ---
def _log_prior(p):
    for v, (lo, hi) in zip(p, PRIOR_BOUNDS):
        if not (lo <= v <= hi): return -np.inf
    return 0.0

def _log_post(p):
    lp = _log_prior(p)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(p[None, :], df_in)[0]
    if not np.isfinite(ll): return -np.inf
    return lp + ll

rng = np.random.default_rng(42)  # SAME seed as our S5 run
centres = np.array([(lo + hi)/2 for lo, hi in PRIOR_BOUNDS])
widths  = np.array([(hi - lo)     for lo, hi in PRIOR_BOUNDS])
n_walkers = 64
p0 = centres + 0.01 * widths * rng.standard_normal((n_walkers, len(PRIOR_BOUNDS)))
for i, (lo, hi) in enumerate(PRIOR_BOUNDS):
    p0[:, i] = np.clip(p0[:, i], lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))

n_steps, n_burn = 8000, 2000
print(f"emcee: {n_walkers} walkers × {n_steps} steps ({n_burn} burn-in)")
t0 = time.time()
sampler = emcee.EnsembleSampler(n_walkers, len(PRIOR_BOUNDS), _log_post)
sampler.run_mcmc(p0, n_steps, progress=True)
print(f"  wall time: {time.time()-t0:.1f} s")
print(f"  mean accept frac: {np.mean(sampler.acceptance_fraction):.3f}")

chain = sampler.get_chain(discard=n_burn, thin=10, flat=True)
samples = pd.DataFrame(chain, columns=PARAM_NAMES)

# --- Summarize ---
print()
print("=" * 70)
print("Posterior medians and 16/84 percentiles (units: km/s, dex, mas/yr)")
print("=" * 70)
def pct(col): return np.percentile(samples[col], [16, 50, 84])
for n in PARAM_NAMES:
    lo, med, hi = pct(n)
    print(f"  {n:18s}  {med:>8.4f}  +{hi-med:>6.4f} -{med-lo:>6.4f}")

# Quantities matching the paper text
v_l, v_m, v_h = pct('vhel_0')
ls_l, ls_m, ls_h = pct('log_sig_vhel')
print()
print(f"Headline:  v_hel = {v_m:.2f}^{{+{v_h-v_m:.2f}}}_{{-{v_m-v_l:.2f}}} km/s")
sig_lo, sig_med, sig_hi = 10**ls_l, 10**ls_m, 10**ls_h
print(f"           sigma_v = {sig_med:.2f}^{{+{sig_hi-sig_med:.2f}}}_{{-{sig_med-sig_lo:.2f}}} km/s")

# Membership
mp = membership_prob(samples.values, df_in)
p_med = np.median(mp, axis=0)
n_99 = int((p_med > 0.99).sum()); n_95 = int((p_med > 0.95).sum())
n_80 = int((p_med > 0.80).sum()); n_50 = int((p_med > 0.50).sum())
print(f"\nMembership counts: >0.99: {n_99}; >0.95: {n_95}; >0.80: {n_80}; >0.50: {n_50}")

# Save outputs
out_h5 = OUT_DIR / "boo3_gmm_geha26_samples.h5"
samples.to_hdf(out_h5, key='samples', mode='w')
print(f"\nWrote {out_h5}")

# Also save per-star membership table
df_in['p_mem'] = p_med
out_csv = OUT_DIR / "boo3_gmm_geha26_membership.csv"
df_in[['source_id','vel_calib','vel_calib_std','feh50','pmra','pmdec','p_mem']].astype('float64', errors='ignore').to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")
