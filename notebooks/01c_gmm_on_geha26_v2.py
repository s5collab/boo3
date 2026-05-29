"""
01c_gmm_on_geha26_v2.py — same GMM (priors + sampler) as our S5 analysis,
applied to Geha+26's Boo III catalog, but the per-star likelihood is modified
to skip the FeH term for stars with no ew_feh measurement and skip the PM
term for stars with no Gaia PM measurement.  This matches the paper text:
"retaining BHB-flagged stars and stars without Gaia proper motions".

Cuts applied:
- |v_hel| < 600                     (VHEL_BOX, RV plausibility)
- 4x4 PM box  --  only enforced for stars that HAVE Gaia PMs
- exclude Gaia DR3 RR Lyrae cross-match (RRL_GEHA)
- v_err finite & > 0
- NO sigma_v < 10 cut
- NO requirement that ew_feh is measured
- NO requirement that Gaia PM is present

Likelihood:
- Galaxy: ll_v + (has_feh ? ll_f : 0) + (has_pm ? ll_pm : 0)
- Background: ll_v_bg + (has_feh ? ll_feh_bg : 0) + (has_pm ? -log(pm_area) : 0)
"""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from astropy import table as atable
import emcee
import importlib.util

warnings.filterwarnings("ignore", category=UserWarning)

NB_DIR = Path(__file__).resolve().parent
PROJ   = NB_DIR.parent
DATA_DIR = PROJ / "data"
OUT_DIR  = PROJ / "output"

# Import constants from the canonical GMM script
spec = importlib.util.spec_from_file_location("compute_gmm", str(NB_DIR / "01_compute_gmm.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PARAM_NAMES   = m.PARAM_NAMES
PRIOR_BOUNDS  = m.PRIOR_BOUNDS
PM_BOX        = m.PM_BOX
VHEL_BOX      = m.VHEL_BOX
FEH_BOX       = m.FEH_BOX
_trunc_norm_logpdf_vec = m._trunc_norm_logpdf_vec

print("=" * 72)
print("01c_gmm_on_geha26_v2.py  — Same priors, missing-data-aware GMM")
print("=" * 72)
print(f"PRIOR_BOUNDS = {PRIOR_BOUNDS}")
print(f"PM_BOX  = {PM_BOX}")
print(f"VHEL_BOX = {VHEL_BOX}, FEH_BOX = {FEH_BOX}")
print()

# --- Load Geha+26 data ---
t = atable.Table.read(DATA_DIR / "geha2026_keck_bootes_3.fits")
RRL_GEHA = 1450796178282259072

v       = np.asarray(t['v'], dtype=float)
v_err   = np.asarray(t['v_err'], dtype=float)
feh     = np.asarray(t['ew_feh'], dtype=float)
feh_err = np.asarray(t['ew_feh_err'], dtype=float)
pmra    = np.asarray(t['gaia_pmra'], dtype=float)
pmdec   = np.asarray(t['gaia_pmdec'], dtype=float)
pmra_e  = np.asarray(t['gaia_pmra_err'], dtype=float)
pmdec_e = np.asarray(t['gaia_pmdec_err'], dtype=float)
rho_pm  = np.asarray(t['gaia_pmra_pmdec_corr'], dtype=float)
sid     = np.asarray(t['gaia_source_id'], dtype=np.int64)

# Per-star availability masks
has_feh = np.isfinite(feh) & (feh > -100) & (feh > FEH_BOX[0]) & (feh < FEH_BOX[1])
has_pm  = (sid != -999) & np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(pmra_e) & np.isfinite(pmdec_e) & (pmra_e > 0) & (pmdec_e > 0)

# Cuts at the catalog level (not in the likelihood)
mask = np.isfinite(v) & np.isfinite(v_err) & (v_err > 0) & (v > VHEL_BOX[0]) & (v < VHEL_BOX[1])
mask &= (sid != RRL_GEHA)
# 4x4 PM box: enforce ONLY for stars that HAVE Gaia PMs
pm_in_box = has_pm & (pmra > PM_BOX[0]) & (pmra < PM_BOX[1]) & (pmdec > PM_BOX[2]) & (pmdec < PM_BOX[3])
mask &= (pm_in_box | ~has_pm)

idx = np.where(mask)[0]
print(f"Catalog rows: {len(t)};  pass initial cuts (v finite, |v|<600, not RRL, PM box if applicable): {len(idx)}")
print(f"  of which: has FeH = {has_feh[idx].sum()},  has PM = {has_pm[idx].sum()}")

# Apply: we also want stars to have Pmem > some level OR not — actually let's
# input the full filtered catalog and let the GMM decide
df = pd.DataFrame({
    'source_id': sid[idx],
    'v': v[idx], 've': v_err[idx],
    'feh': feh[idx], 'feh_err': feh_err[idx],
    'pmra': pmra[idx], 'pmdec': pmdec[idx],
    'pmra_e': pmra_e[idx], 'pmdec_e': pmdec_e[idx],
    'rho_pm': rho_pm[idx],
    'has_feh': has_feh[idx], 'has_pm': has_pm[idx],
})

# Also report Pmem_novar > 0.5 overlap
pmem_novar = np.asarray(t['Pmem_novar'])[idx]
print(f"  of which: in Geha Pmem_novar>0.5 = {(pmem_novar > 0.5).sum()}")
N = len(df)
print(f"GMM input N = {N}\n")

# Replace missing PMs and FeH with finite but safe values so vectorized math
# doesn't choke — but the gating mask will zero out their contribution.
def _safe(arr, fill):
    out = arr.copy().astype(float)
    out[~np.isfinite(out)] = fill
    return out
v_arr   = df['v'].values
ve_arr  = df['ve'].values
fe_arr  = _safe(df['feh'].values, 0.0)
fee_arr = _safe(df['feh_err'].values, 1.0)
pa_arr  = _safe(df['pmra'].values, 0.0)
pd_arr  = _safe(df['pmdec'].values, 0.0)
sa_arr  = _safe(df['pmra_e'].values, 1.0)
sd_arr  = _safe(df['pmdec_e'].values, 1.0)
rho_arr = _safe(df['rho_pm'].values, 0.0)
has_feh_arr = df['has_feh'].values.astype(float)
has_pm_arr  = df['has_pm'].values.astype(float)

def _gal_ll(p):
    """Galaxy log-likelihood vector over N stars; p shape (1, 11)."""
    vhel_0  = p[:, 1:2];  log_sig_vhel = p[:, 2:3]
    feh_0   = p[:, 3:4];  log_sig_feh  = p[:, 4:5]
    pmr_0   = p[:, 5:6];  pmd_0        = p[:, 6:7]
    sig_v   = 10 ** log_sig_vhel
    sig_feh = 10 ** log_sig_feh
    v0  = v_arr[None, :];  ve  = ve_arr[None, :]
    fe  = fe_arr[None, :];  fee = fee_arr[None, :]
    pa  = pa_arr[None, :];  pd_ = pd_arr[None, :]
    sa  = sa_arr[None, :];  sd  = sd_arr[None, :]
    rho = rho_arr[None, :]
    # v term — ALWAYS present
    sv2  = ve**2 + sig_v**2
    ll_v = -0.5 * np.log(2*np.pi*sv2) - 0.5*(v0 - vhel_0)**2 / sv2
    # FeH term — only where has_feh
    sf2  = fee**2 + sig_feh**2
    ll_f = -0.5 * np.log(2*np.pi*sf2) - 0.5*(fe - feh_0)**2 / sf2
    ll_f *= has_feh_arr[None, :]
    # PM term — only where has_pm
    cov_xy = rho * sa * sd
    var_x = sa**2; var_y = sd**2
    det = var_x*var_y - cov_xy**2
    dx = pa - pmr_0;  dy = pd_ - pmd_0
    quad = (dy**2*var_x - 2*dx*dy*cov_xy + dx**2*var_y) / det
    ll_pm = -np.log(2*np.pi) - 0.5*np.log(det) - 0.5*quad
    ll_pm *= has_pm_arr[None, :]
    return ll_v + ll_f + ll_pm

def _back_ll(p):
    """Background log-likelihood vector."""
    bg_v_mean = p[:, 7:8]; log_bg_v_sigma = p[:, 8:9]
    bg_feh_mean = p[:, 9:10]; log_bg_feh_sigma = p[:, 10:11]
    bg_v_sigma   = 10 ** log_bg_v_sigma
    bg_feh_sigma = 10 ** log_bg_feh_sigma
    v0 = v_arr[None, :];  fe = fe_arr[None, :]
    ll_v_bg = _trunc_norm_logpdf_vec(v0, bg_v_mean, bg_v_sigma, VHEL_BOX[0], VHEL_BOX[1])
    ll_feh_bg = _trunc_norm_logpdf_vec(fe, bg_feh_mean, bg_feh_sigma, FEH_BOX[0], FEH_BOX[1])
    ll_feh_bg *= has_feh_arr[None, :]
    pm_area = (PM_BOX[1]-PM_BOX[0]) * (PM_BOX[3]-PM_BOX[2])
    ll_pm_bg = -np.log(pm_area) * has_pm_arr[None, :]
    return ll_v_bg + ll_feh_bg + ll_pm_bg

def log_likelihood(p):
    if p.ndim == 1: p = p[None, :]
    f_mem = p[:, 0:1]
    mix = np.logaddexp(np.log(f_mem) + _gal_ll(p),
                        np.log(1.0 - f_mem) + _back_ll(p))
    return mix.sum(axis=1)

def membership_prob(p):
    if p.ndim == 1: p = p[None, :]
    f_mem = p[:, 0:1]
    gl = np.exp(_gal_ll(p))
    bl = np.exp(_back_ll(p))
    return (f_mem * gl) / (f_mem * gl + (1.0 - f_mem) * bl)

def _log_prior(p):
    for v_, (lo, hi) in zip(p, PRIOR_BOUNDS):
        if not (lo <= v_ <= hi): return -np.inf
    return 0.0

def _log_post(p):
    lp = _log_prior(p)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(p[None, :])[0]
    if not np.isfinite(ll): return -np.inf
    return lp + ll

# --- Run emcee, same settings as the S5 GMM ---
rng = np.random.default_rng(42)
centres = np.array([(lo + hi)/2 for lo, hi in PRIOR_BOUNDS])
widths  = np.array([(hi - lo)     for lo, hi in PRIOR_BOUNDS])
n_walkers = 64
p0 = centres + 0.01 * widths * rng.standard_normal((n_walkers, len(PRIOR_BOUNDS)))
for i, (lo, hi) in enumerate(PRIOR_BOUNDS):
    p0[:, i] = np.clip(p0[:, i], lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
n_steps, n_burn = 8000, 2000
print(f"emcee: {n_walkers} walkers x {n_steps} steps ({n_burn} burn-in)")
t0 = time.time()
sampler = emcee.EnsembleSampler(n_walkers, len(PRIOR_BOUNDS), _log_post)
sampler.run_mcmc(p0, n_steps, progress=True)
print(f"  wall: {time.time()-t0:.1f}s   accept frac: {np.mean(sampler.acceptance_fraction):.3f}")
chain = sampler.get_chain(discard=n_burn, thin=10, flat=True)
samples = pd.DataFrame(chain, columns=PARAM_NAMES)

# --- Summarize ---
print()
print("=" * 72)
print("Posterior (median +/- 1sigma):")
print("=" * 72)
def pct(c): return np.percentile(samples[c], [16, 50, 84])
for n in PARAM_NAMES:
    lo, med, hi = pct(n)
    print(f"  {n:18s}  {med:>8.4f}  +{hi-med:>6.4f} -{med-lo:>6.4f}")

v_l, v_m, v_h = pct('vhel_0')
ls_l, ls_m, ls_h = pct('log_sig_vhel')
sig_lo, sig_med, sig_hi = 10**ls_l, 10**ls_m, 10**ls_h
print()
print(f"Headline:  v_hel   = {v_m:.2f}^{{+{v_h-v_m:.2f}}}_{{-{v_m-v_l:.2f}}} km/s")
print(f"           sigma_v = {sig_med:.2f}^{{+{sig_hi-sig_med:.2f}}}_{{-{sig_med-sig_lo:.2f}}} km/s")

# Membership probabilities
mp = membership_prob(samples.values)
p_med = np.median(mp, axis=0)
df['p_mem'] = p_med
print(f"\nMembership: >0.99: {int((p_med>0.99).sum())};  >0.95: {int((p_med>0.95).sum())};  >0.80: {int((p_med>0.80).sum())};  >0.50: {int((p_med>0.50).sum())}")

# How many of Geha's 16 we recovered as members
pmem_novar_arr = np.asarray(t['Pmem_novar'])[idx]
gehas16 = pmem_novar_arr > 0.5
recovered = int(((p_med > 0.5) & gehas16).sum())
total_gehas16 = int(gehas16.sum())
print(f"  of Geha's 16 σ_v sample: {recovered}/{total_gehas16} have our p_mem>0.5")

# Save
out_h5 = OUT_DIR / "boo3_gmm_geha26v2_samples.h5"
samples.to_hdf(out_h5, key='samples', mode='w')
print(f"\nWrote {out_h5}")
out_csv = OUT_DIR / "boo3_gmm_geha26v2_membership.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")
