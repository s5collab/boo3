"""
01_compute_gmm.py — heavy compute for notebook 01.

Run BEFORE `01_members_gmm_orbits.ipynb` (the notebook only loads + plots).

What it does:
  1. Builds the 120-star input catalog from S5 DR2 with the new
     ellipse-centroid (RA = 209.5567, Dec = 26.5529).  Writes
     `output/boo3_input_127.csv` (filename kept for back-compat).
  2. Runs the 11-parameter GMM mixture model with emcee (64 walkers x
     8000 steps, 2000 burn-in).  Writes `output/boo3_gmm_samples.h5`.
  3. Computes per-star membership probabilities and writes
     `output/boo3_gmm_membership.csv`.

These three outputs are the only inputs the notebook needs from this script.
Re-run only if the input catalog or the priors change.
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

# ----- Project paths --------------------------------------------------------
NB_DIR   = Path(__file__).resolve().parent
PROJ     = NB_DIR.parent
DATA_DIR = PROJ / "data"
OUT_DIR  = PROJ / "output"
OUT_DIR.mkdir(exist_ok=True)

S5_DR2_FITS = DATA_DIR / "cat_s5_public_dr2.0_beta0.fits"
RRL_CSV     = DATA_DIR / "gaiadr3_RRLyrae_boo3_5deg.csv"

INPUT_CSV   = OUT_DIR / "boo3_input_127.csv"
SAMPLES_H5  = OUT_DIR / "boo3_gmm_samples.h5"
MEMSHIP_CSV = OUT_DIR / "boo3_gmm_membership.csv"

# ----- Boo III geometry (also defined in nb01 cell 4, kept in sync) --------
RA_BOO3        = 209.5567       # deg  (NEW ellipse-centroid; this work)
DEC_BOO3       = 26.5529        # deg
ELLIPTICITY    = 0.33
PA_DEG         = 278.91         # deg E of N
RHALF_CIRC_AM  = 33.03          # arcmin (azimuthally averaged, Moskowitz+20)
RHALF_MAJOR_AM = RHALF_CIRC_AM / np.sqrt(1 - ELLIPTICITY)
RH3_DEG        = 3 * RHALF_MAJOR_AM / 60.0

# ----- Background boxes ----------------------------------------------------
PM_CENTRE = (-1.17, -0.88)
PM_HALF   = 2.0
PM_BOX    = (PM_CENTRE[0]-PM_HALF, PM_CENTRE[0]+PM_HALF,
             PM_CENTRE[1]-PM_HALF, PM_CENTRE[1]+PM_HALF)
VHEL_BOX  = (-600.0, 600.0)
FEH_BOX   = (-5.0,    0.0)


# ===========================================================================
# 1. Catalog utilities (mirror cell 6 of nb01)
# ===========================================================================
def deredden_mag_S5(table):
    g = table["decam_g"] - 3.185 * table["ebv"]
    r = table["decam_r"] - 2.140 * table["ebv"]
    i = table["decam_i"] - 1.569 * table["ebv"]
    z = table["decam_z"] - 1.196 * table["ebv"]
    return g, r, i, z


def add_delta_coordinates(table, ra0, dec0):
    ra0_r = np.radians(ra0); dec0_r = np.radians(dec0)
    ra_r  = np.radians(table["ra"]); dec_r = np.radians(table["dec"])
    dra_r = np.cos(dec_r) * np.sin(ra_r - ra0_r)
    ddec_r = (np.sin(dec_r) * np.cos(dec0_r)
              - np.cos(dec_r) * np.sin(dec0_r) * np.cos(ra_r - ra0_r))
    table["ra_delta"]  = np.degrees(dra_r)
    table["dec_delta"] = np.degrees(ddec_r)


def ellipse_radius(ra_delta, dec_delta, ellipticity=ELLIPTICITY, pa_deg=PA_DEG):
    q   = 1 - ellipticity
    ang = np.radians(90 - pa_deg)
    ra_r  =  ra_delta * np.cos(ang) + dec_delta * np.sin(ang)
    dec_r = -ra_delta * np.sin(ang) + dec_delta * np.cos(ang)
    return np.sqrt(ra_r**2 + dec_r**2 / q**2)


# ===========================================================================
# 2. Build 120-star input catalog (mirror cell 8 of nb01)
# ===========================================================================
def build_input_catalog():
    print(f"  S5 DR2:  {S5_DR2_FITS}")
    with fits.open(str(S5_DR2_FITS)) as h:
        t = atable.Table(h[1].data)
    print(f"  total S5 DR2 rows:        {len(t):>7}")
    t = t[t["object_name"] == "Styx"]
    print(f"  Boötes III field (Styx):  {len(t):>7}")

    t["decam_g0"], t["decam_r0"], *_ = deredden_mag_S5(t)
    add_delta_coordinates(t, RA_BOO3, DEC_BOO3)

    keep = ((t["best_sn_1700d"] > 2)
            & (t["vel_calib_std"] < 10)
            & (t["good_star_pb"]  > 0.5)
            & np.isfinite(t["pmra"]) & np.isfinite(t["pmdec"])
            & (np.abs(t["vel_calib"]) < 600)
            & (t["feh50"] < 0))
    t = t[keep]
    print(f"  after quality cuts:       {len(t):>7}")

    r_ell = ellipse_radius(np.asarray(t["ra_delta"]), np.asarray(t["dec_delta"]))
    t = t[r_ell < RH3_DEG]
    print(f"  inside 3 r_h ellipse:     {len(t):>7}")

    rrl_df = pd.read_csv(str(RRL_CSV))
    cr = SkyCoord(ra=rrl_df["ra"].values * u.deg, dec=rrl_df["dec"].values * u.deg)
    cs = SkyCoord(ra=t["ra"].data * u.deg, dec=t["dec"].data * u.deg)
    _, sep, _ = cs.match_to_catalog_sky(cr)
    n_rrl = int((sep < 2 * u.arcsec).sum())
    t = t[sep >= 2 * u.arcsec]
    print(f"  RRL matches removed:      {n_rrl:>7}")
    print(f"  non-RRL stars in 3 r_h:   {len(t):>7}")

    in_pm = ((t["pmra"]  > PM_BOX[0]) & (t["pmra"]  < PM_BOX[1])
             & (t["pmdec"] > PM_BOX[2]) & (t["pmdec"] < PM_BOX[3]))
    t = t[in_pm]
    print(f"  inside 4x4 PM box:        {len(t):>7}")
    assert len(t) == 120, f"Expected 120 stars in NEW (ellipse-centroid) input, got {len(t)}"

    df = t.to_pandas()
    sid_col = next(c for c in ("source_id", "gaia_source_id", "gaia_id") if c in df.columns)
    out = pd.DataFrame({
        "source_id":       df[sid_col].astype("int64").values,
        "ra":              df["ra"].values,
        "dec":             df["dec"].values,
        "ra_delta":        df["ra_delta"].values,
        "dec_delta":       df["dec_delta"].values,
        "decam_g0":        df["decam_g0"].values,
        "decam_r0":        df["decam_r0"].values,
        "vel_calib":       df["vel_calib"].values,
        "vel_calib_std":   df["vel_calib_std"].values,
        "feh50":           df["feh50"].values,
        "feh_calib_std":   df["feh_calib_std"].values,
        "pmra":            df["pmra"].values,
        "pmdec":           df["pmdec"].values,
        "pmra_error":      df["pmra_error"].values,
        "pmdec_error":     df["pmdec_error"].values,
        "pmra_pmdec_corr": df["pmra_pmdec_corr"].values,
        "best_sn_1700d":   df["best_sn_1700d"].values,
        "good_star_pb":    df["good_star_pb"].values,
    })
    out.to_csv(INPUT_CSV, index=False)
    print(f"  wrote {INPUT_CSV}  ({len(out)} stars)")
    return out


# ===========================================================================
# 3. GMM model: priors + likelihood (mirror cell 11 of nb01)
# ===========================================================================
PARAM_NAMES = [
    "f_mem", "vhel_0", "log_sig_vhel", "feh_0", "log_sig_feh",
    "pmr_0", "pmd_0",
    "bg_v_mean", "log_bg_v_sigma", "bg_feh_mean", "log_bg_feh_sigma",
]
PRIOR_BOUNDS = [
    (0.0, 1.0), (170.0, 210.0), (-2.0, 1.3), (-3.5, -1.2), (-2.0, 0.3),
    (-2.0, -0.5), (-1.5, 0.0),
    (-600.0, 600.0), (1.0, 2.5), (-5.0, 0.0), (-1.0, 0.3),
]


def _gal_log_likelihood(p, obs):
    vhel_0 = p[:, 1:2]; log_sig_vhel = p[:, 2:3]
    feh_0  = p[:, 3:4]; log_sig_feh  = p[:, 4:5]
    pmr_0  = p[:, 5:6]; pmd_0        = p[:, 6:7]
    sig_v   = 10 ** log_sig_vhel
    sig_feh = 10 ** log_sig_feh

    v   = obs["vel_calib"].values[None, :];      ve  = obs["vel_calib_std"].values[None, :]
    fe  = obs["feh50"].values[None, :];          fee = obs["feh_calib_std"].values[None, :]
    pa  = obs["pmra"].values[None, :];           pd_ = obs["pmdec"].values[None, :]
    sa  = obs["pmra_error"].values[None, :];     sd  = obs["pmdec_error"].values[None, :]
    rho = obs["pmra_pmdec_corr"].values[None, :]

    sv2  = ve**2 + sig_v**2
    ll_v = -0.5 * np.log(2*np.pi*sv2) - 0.5 * (v - vhel_0)**2 / sv2

    sf2  = fee**2 + sig_feh**2
    ll_f = -0.5 * np.log(2*np.pi*sf2) - 0.5 * (fe - feh_0)**2 / sf2

    cov_xy = rho * sa * sd
    var_x  = sa**2; var_y = sd**2
    det    = var_x * var_y - cov_xy**2
    dx = pa - pmr_0; dy = pd_ - pmd_0
    quad = (dy**2 * var_x - 2 * dx * dy * cov_xy + dx**2 * var_y) / det
    ll_pm = -np.log(2*np.pi) - 0.5*np.log(det) - 0.5 * quad
    return ll_v + ll_f + ll_pm


def _trunc_norm_logpdf_vec(x, mu, sigma, low, high):
    log_pdf  = stats.norm.logpdf(x, loc=mu, scale=sigma)
    log_norm = np.log(stats.norm.cdf(high, loc=mu, scale=sigma)
                      - stats.norm.cdf(low,  loc=mu, scale=sigma))
    return log_pdf - log_norm


def _back_log_likelihood(p, obs):
    bg_v_mean        = p[:, 7:8];  log_bg_v_sigma   = p[:, 8:9]
    bg_feh_mean      = p[:, 9:10]; log_bg_feh_sigma = p[:, 10:11]
    bg_v_sigma   = 10 ** log_bg_v_sigma
    bg_feh_sigma = 10 ** log_bg_feh_sigma
    v  = obs["vel_calib"].values[None, :]
    fe = obs["feh50"].values[None, :]
    ll_v_bg   = _trunc_norm_logpdf_vec(v, bg_v_mean,  bg_v_sigma,
                                       VHEL_BOX[0], VHEL_BOX[1])
    ll_feh_bg = _trunc_norm_logpdf_vec(fe, bg_feh_mean, bg_feh_sigma,
                                       FEH_BOX[0], FEH_BOX[1])
    pm_area  = (PM_BOX[1]-PM_BOX[0]) * (PM_BOX[3]-PM_BOX[2])
    ll_pm_bg = -np.log(pm_area)
    return ll_v_bg + ll_feh_bg + ll_pm_bg


def log_likelihood(p, obs):
    if p.ndim == 1:
        p = p[None, :]
    f_mem = p[:, 0:1]
    gal_ll  = _gal_log_likelihood(p, obs)
    back_ll = _back_log_likelihood(p, obs)
    mix = np.logaddexp(np.log(f_mem) + gal_ll,
                        np.log(1.0 - f_mem) + back_ll)
    return mix.sum(axis=1)


def membership_prob(p, obs):
    if p.ndim == 1:
        p = p[None, :]
    f_mem = p[:, 0:1]
    gal_like  = np.exp(_gal_log_likelihood(p, obs))
    back_like = np.exp(_back_log_likelihood(p, obs))
    num = f_mem * gal_like
    den = num + (1.0 - f_mem) * back_like
    return num / den


# ===========================================================================
# 4. emcee driver (mirror cell 14 of nb01)
# ===========================================================================
def run_emcee(df_in, n_walkers=64, n_steps=8000, n_burn=2000, seed=42):
    def _log_prior(p):
        for v, (lo, hi) in zip(p, PRIOR_BOUNDS):
            if not (lo <= v <= hi):
                return -np.inf
        return 0.0

    def _log_post(p):
        lp = _log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(p[None, :], df_in)[0]
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    rng = np.random.default_rng(seed)
    centres = np.array([(lo + hi) / 2 for lo, hi in PRIOR_BOUNDS])
    widths  = np.array([(hi - lo)     for lo, hi in PRIOR_BOUNDS])
    p0 = centres + 0.01 * widths * rng.standard_normal((n_walkers, len(PRIOR_BOUNDS)))
    for i, (lo, hi) in enumerate(PRIOR_BOUNDS):
        p0[:, i] = np.clip(p0[:, i], lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))

    print(f"Running emcee: {n_walkers} walkers x {n_steps} steps "
          f"({n_burn} burn-in, {len(df_in)} stars) ...")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(n_walkers, len(PRIOR_BOUNDS), _log_post)
    sampler.run_mcmc(p0, n_steps, progress=False)
    print(f"  emcee wall time: {time.time() - t0:.1f} s")
    print(f"  mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    chain = sampler.get_chain(discard=n_burn, thin=10, flat=True)
    print(f"  flat chain shape: {chain.shape}")
    return pd.DataFrame(chain, columns=PARAM_NAMES)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("01_compute_gmm.py — building 120-star input catalog")
    print("=" * 70)
    df_in = build_input_catalog()

    print()
    print("=" * 70)
    print("Sanity check at literature systemic point")
    print("=" * 70)
    test_par = np.array([[0.16, 191.4, np.log10(1.1), -2.34, np.log10(0.29),
                          -1.17, -0.88, 0.0, 100.0, -1.3, 0.6]])
    print(f"log-likelihood at literature point: {log_likelihood(test_par, df_in)[0]:.3f}")

    print()
    print("=" * 70)
    print("Running emcee GMM sampler")
    print("=" * 70)
    samples = run_emcee(df_in)
    samples.to_hdf(SAMPLES_H5, key="samples", mode="w")
    print(f"  saved {SAMPLES_H5}")

    print()
    print("=" * 70)
    print("Per-star membership probabilities")
    print("=" * 70)
    mp_samp = membership_prob(samples.values, df_in)
    df_in["p_mem"]    = np.median(mp_samp, axis=0)
    df_in["p_mem_lo"] = np.percentile(mp_samp, 16, axis=0)
    df_in["p_mem_hi"] = np.percentile(mp_samp, 84, axis=0)
    n_99 = int((df_in["p_mem"] > 0.99).sum())
    n_95 = int((df_in["p_mem"] > 0.95).sum())
    n_80 = int((df_in["p_mem"] > 0.80).sum())
    n_50 = int((df_in["p_mem"] > 0.50).sum())
    print(f"  p_mem > 0.99: {n_99}")
    print(f"  p_mem > 0.95: {n_95}")
    print(f"  p_mem > 0.80: {n_80}")
    print(f"  p_mem > 0.50: {n_50}")
    df_in.to_csv(MEMSHIP_CSV, index=False)
    print(f"  wrote {MEMSHIP_CSV}")

    print()
    print("Done.  Notebook 01 can now read the three artefacts:")
    print(f"  - {INPUT_CSV.name}")
    print(f"  - {SAMPLES_H5.name}")
    print(f"  - {MEMSHIP_CSV.name}")


if __name__ == "__main__":
    main()
