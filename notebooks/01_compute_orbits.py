"""
01_compute_orbits.py — heavy orbital integrations for notebook 01.

Run AFTER `01_compute_gmm.py` and BEFORE `01_members_gmm_orbits.ipynb`.

What it does (Section 4 + Appendix A.1 + A.3 of the paper):
  1. McMillan17 fid + 1000-MC, backward 4 Gyr, with and without LMC.
     -> output/orbit_McMillan17_trajectories.npz   (for Figure 3)
     -> output/orbit_McMillan17_samples.npz        (peri/apo/e for Figure 4)
  2. MWPotential2014 fid + 1000-MC, backward 4 Gyr, with and without LMC.
     -> output/orbit_MWPotential2014_trajectories.npz   (for Figure A1)
     -> output/orbit_MWPotential2014_samples.npz        (peri/apo/e for Figure A2)
  3. Gradient-MC: 1000 realisations, +/- 0.5 Gyr forward+backward, McMillan17+LMC.
     -> output/orbit_gradient_mc_1000_summary.npz  (for Figure 7 error bar)
  4. Per-component MC: 100 realisations x 6 components, +/- 0.5 Gyr.
     -> output/orbit_per_component_trajectories.npz   (for Figure A4)
     -> output/orbit_per_component_summary.npz        (gE/gN summary)

The notebook only loads these npz files — no orbit integration in the notebook.
Re-run only if the input GMM samples or the 6D constants change.
"""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
from scipy.signal import argrelextrema

from galpy.orbit import Orbit
from galpy.util import galpyWarning
from galpy.util.conversion import get_physical
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce, HernquistPotential,
    MovingObjectPotential, NonInertialFrameForce,
    evaluateRforces, evaluatephitorques, evaluatezforces,
    MWPotential2014,
)
from galpy.potential.mwpotentials import McMillan17

warnings.filterwarnings("ignore", category=galpyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Project paths --------------------------------------------------------
NB_DIR  = Path(__file__).resolve().parent
PROJ    = NB_DIR.parent
OUT_DIR = PROJ / "output"
OUT_DIR.mkdir(exist_ok=True)
SAMPLES_H5 = OUT_DIR / "boo3_gmm_samples.h5"

# ---- Boo III geometry (kept in sync with 01_compute_gmm.py + nb01) -------
RA_BOO3, DEC_BOO3   = 209.5567, 26.5529
RA_BOO3_ERR         = 0.3
DEC_BOO3_ERR        = 0.3
DIST_KPC, DIST_ERR  = 48.47, 1.9  # 5-RRL G22 mean

# ---- Run knobs ------------------------------------------------------------
SOLAR_MOTION = [-12.9, 12.5, 7.78]
N_REAL = 1000     # MC realisations for Fig 3 / Fig A1 / Fig 7 error bar
T_GYR  = 4.0      # backward integration baseline (4 Gyr; LMC encounter at ~3.4 Gyr)
N_STEP = 1001
N_PC      = 100
T_GYR_PC  = 0.5
N_STEP_PC = 1001
COMP_KEYS = ["ra", "dec", "dist", "pmra", "pmdec", "vhel"]
COMP_IDX  = dict(ra=0, dec=1, dist=2, pmra=3, pmdec=4, vhel=5)


# ===========================================================================
# Load 6D from GMM
# ===========================================================================
samples = pd.read_hdf(SAMPLES_H5, key="samples")
vhel_med  = float(np.median(samples["vhel_0"]));  vhel_err  = float(np.std(samples["vhel_0"]))
pmra_med  = float(np.median(samples["pmr_0"]));   pmra_err  = float(np.std(samples["pmr_0"]))
pmdec_med = float(np.median(samples["pmd_0"]));   pmdec_err = float(np.std(samples["pmd_0"]))

PARAMS_6D = [RA_BOO3, DEC_BOO3, DIST_KPC, pmra_med, pmdec_med, vhel_med]
ERRORS_6D = [RA_BOO3_ERR, DEC_BOO3_ERR, DIST_ERR, pmra_err, pmdec_err, vhel_err]

print("=" * 70)
print("01_compute_orbits.py")
print("=" * 70)
print(f"systemic centre:  v_hel = {vhel_med:+.2f} +/- {vhel_err:.2f} km/s")
print(f"                  pmra  = {pmra_med:+.4f} +/- {pmra_err:.4f} mas/yr")
print(f"                  pmdec = {pmdec_med:+.4f} +/- {pmdec_err:.4f} mas/yr")
print(f"                  d     = {DIST_KPC} +/- {DIST_ERR} kpc")
print(f"                  RA    = {RA_BOO3:.4f} +/- {RA_BOO3_ERR} deg")
print(f"                  Dec   = {DEC_BOO3:.4f} +/- {DEC_BOO3_ERR} deg")


# ===========================================================================
# Helpers
# ===========================================================================
def build_total_potential_backward(base_pot):
    """MW + LMC (with dyn. friction) + non-inertial frame correction."""
    m_lmc = 1.38e11; rs = 16.09; rhm = rs * (1 + np.sqrt(2))
    orb_lmc = Orbit.from_name("LMC", solarmotion=SOLAR_MOTION,
                              **get_physical(base_pot))
    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=m_lmc * u.Msun, rhm=rhm * u.kpc, dens=base_pot[1],
        **get_physical(base_pot))
    ts = np.linspace(0, 5, 1001) * u.Gyr
    orb_lmc.integrate(-ts, base_pot + [cdf])

    lmcpot = HernquistPotential(amp=2 * m_lmc * u.Msun, a=rs * u.kpc,
                                 **get_physical(base_pot))
    moving = MovingObjectPotential(orb_lmc, pot=lmcpot,
                                    **get_physical(base_pot))
    loc = 1e-4
    af = lambda t: evaluateRforces(moving, loc, 0., phi=0., t=t, use_physical=False)
    bf = lambda t: evaluatephitorques(moving, loc, 0., phi=0., t=t, use_physical=False) / loc
    cf = lambda t: evaluatezforces(moving, loc, 0., phi=0., t=t, use_physical=False)
    ti = orb_lmc.time(use_physical=False)[::-1]
    aa = np.array([af(t) for t in ti])
    ab = np.array([bf(t) for t in ti])
    ac = np.array([cf(t) for t in ti])
    nip = NonInertialFrameForce(a0=[lambda t: np.interp(t, ti, aa),
                                     lambda t: np.interp(t, ti, ab),
                                     lambda t: np.interp(t, ti, ac)])
    return base_pot + [nip, moving]


def integrate_backward(pot_bwd, base_pot, label):
    print(f"\n-> {label}, {N_REAL} realisations, {T_GYR} Gyr ...")
    ts = np.linspace(0, T_GYR, N_STEP) * u.Gyr
    rng = np.random.default_rng(2024)
    p6d_many = rng.normal(loc=PARAMS_6D, scale=ERRORS_6D, size=(N_REAL, 6))
    orb_b   = Orbit(PARAMS_6D, solarmotion=SOLAR_MOTION, radec=True,
                    **get_physical(base_pot))
    orb_b_m = Orbit(p6d_many,  solarmotion=SOLAR_MOTION, radec=True,
                    **get_physical(base_pot))
    orb_b.integrate(-ts, pot_bwd)
    orb_b_m.integrate(-ts, pot_bwd)
    return dict(ts_gyr=ts.to(u.Gyr).value,
                fid_x=np.asarray(orb_b.x(-ts)),
                fid_y=np.asarray(orb_b.y(-ts)),
                fid_z=np.asarray(orb_b.z(-ts)),
                fid_r=np.asarray(orb_b.r(-ts)),
                mc_x=np.asarray(orb_b_m.x(-ts)),
                mc_y=np.asarray(orb_b_m.y(-ts)),
                mc_z=np.asarray(orb_b_m.z(-ts)),
                mc_r=np.asarray(orb_b_m.r(-ts)))


def compute_peri_apo(res):
    """Peri/apo/e from MC trajectories.  res from integrate_backward()."""
    r_all = res["mc_r"]   # shape (n_real, n_step)
    peri, apo = [], []
    for r_ in r_all:
        i_p = argrelextrema(r_, np.less)[0]
        i_a = argrelextrema(r_, np.greater)[0]
        peri.append(r_[i_p[0]] if len(i_p) else np.nan)
        apo.append(r_[i_a[0]] if len(i_a) else np.nan)
    peri = np.array(peri); apo = np.array(apo)
    e = (apo - peri) / (apo + peri)
    return peri, apo, e


def _q(x):
    x = x[np.isfinite(x)]
    lo, m, hi = np.percentile(x, [16, 50, 84])
    return m, m - lo, hi - m


def print_summary(label, peri, apo, e):
    pm, pl, ph = _q(peri); am, al, ah = _q(apo); em, el, eh = _q(e)
    print(f"  {label:<10} peri = {pm:6.2f}  +{ph:.2f} / -{pl:.2f} kpc")
    print(f"  {' ':<10} apo  = {am:6.2f}  +{ah:.2f} / -{al:.2f} kpc")
    print(f"  {' ':<10} e    = {em:6.3f}  +{eh:.3f} / -{el:.3f}")


# ===========================================================================
# 1. McMillan17 (Section 4 — Fig 3 + Fig 4)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 1: McMillan17 (no LMC + with LMC), 4 Gyr backward")
print("=" * 70)
res_no = integrate_backward(McMillan17, McMillan17, "McMillan17, no LMC")
print("\nWith LMC -- building backward LMC potential ...")
pot_bwd_mcm = build_total_potential_backward([p for p in McMillan17])
res_lmc = integrate_backward(pot_bwd_mcm, McMillan17, "McMillan17 + LMC")

np.savez(OUT_DIR / "orbit_McMillan17_trajectories.npz",
         no_lmc_ts=res_no["ts_gyr"],
         no_lmc_fid_x=res_no["fid_x"], no_lmc_fid_y=res_no["fid_y"],
         no_lmc_fid_z=res_no["fid_z"], no_lmc_fid_r=res_no["fid_r"],
         no_lmc_mc_x=res_no["mc_x"],   no_lmc_mc_y=res_no["mc_y"],
         no_lmc_mc_z=res_no["mc_z"],   no_lmc_mc_r=res_no["mc_r"],
         lmc_ts=res_lmc["ts_gyr"],
         lmc_fid_x=res_lmc["fid_x"], lmc_fid_y=res_lmc["fid_y"],
         lmc_fid_z=res_lmc["fid_z"], lmc_fid_r=res_lmc["fid_r"],
         lmc_mc_x=res_lmc["mc_x"],   lmc_mc_y=res_lmc["mc_y"],
         lmc_mc_z=res_lmc["mc_z"],   lmc_mc_r=res_lmc["mc_r"],
         params_6D=PARAMS_6D, errors_6D=ERRORS_6D, n_real=N_REAL, t_gyr=T_GYR)
print(f"  saved {OUT_DIR / 'orbit_McMillan17_trajectories.npz'}")

peri_no,  apo_no,  e_no  = compute_peri_apo(res_no)
peri_lmc, apo_lmc, e_lmc = compute_peri_apo(res_lmc)
np.savez(OUT_DIR / "orbit_McMillan17_samples.npz",
         peri_no=peri_no, apo_no=apo_no, e_no=e_no,
         peri_lmc=peri_lmc, apo_lmc=apo_lmc, e_lmc=e_lmc,
         params_6D=PARAMS_6D, errors_6D=ERRORS_6D)
print(f"  saved {OUT_DIR / 'orbit_McMillan17_samples.npz'}")
print("\nMcMillan17 summary  (median +sig68 / -sig68):")
print_summary("No LMC",   peri_no,  apo_no,  e_no)
print_summary("With LMC", peri_lmc, apo_lmc, e_lmc)


# ===========================================================================
# 2. MWPotential2014 (Appendix A.1 — Fig A1 + Fig A2)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 2: MWPotential2014 (no LMC + with LMC), 4 Gyr backward")
print("=" * 70)
mwp14 = [p for p in MWPotential2014]
res_no_v2 = integrate_backward(mwp14, mwp14, "MWPotential2014, no LMC")
print("\nWith LMC -- building backward LMC potential (MWP14 base) ...")
pot_bwd_v2 = build_total_potential_backward(mwp14)
res_lmc_v2 = integrate_backward(pot_bwd_v2, mwp14, "MWPotential2014 + LMC")

np.savez(OUT_DIR / "orbit_MWPotential2014_trajectories.npz",
         no_lmc_ts=res_no_v2["ts_gyr"],
         no_lmc_fid_x=res_no_v2["fid_x"], no_lmc_fid_y=res_no_v2["fid_y"],
         no_lmc_fid_z=res_no_v2["fid_z"], no_lmc_fid_r=res_no_v2["fid_r"],
         no_lmc_mc_x=res_no_v2["mc_x"],   no_lmc_mc_y=res_no_v2["mc_y"],
         no_lmc_mc_z=res_no_v2["mc_z"],   no_lmc_mc_r=res_no_v2["mc_r"],
         lmc_ts=res_lmc_v2["ts_gyr"],
         lmc_fid_x=res_lmc_v2["fid_x"], lmc_fid_y=res_lmc_v2["fid_y"],
         lmc_fid_z=res_lmc_v2["fid_z"], lmc_fid_r=res_lmc_v2["fid_r"],
         lmc_mc_x=res_lmc_v2["mc_x"],   lmc_mc_y=res_lmc_v2["mc_y"],
         lmc_mc_z=res_lmc_v2["mc_z"],   lmc_mc_r=res_lmc_v2["mc_r"],
         params_6D=PARAMS_6D, errors_6D=ERRORS_6D, n_real=N_REAL, t_gyr=T_GYR)
print(f"  saved {OUT_DIR / 'orbit_MWPotential2014_trajectories.npz'}")

peri_no_v2,  apo_no_v2,  e_no_v2  = compute_peri_apo(res_no_v2)
peri_lmc_v2, apo_lmc_v2, e_lmc_v2 = compute_peri_apo(res_lmc_v2)
np.savez(OUT_DIR / "orbit_MWPotential2014_samples.npz",
         peri_no=peri_no_v2, apo_no=apo_no_v2, e_no=e_no_v2,
         peri_lmc=peri_lmc_v2, apo_lmc=apo_lmc_v2, e_lmc=e_lmc_v2,
         params_6D=PARAMS_6D, errors_6D=ERRORS_6D)
print(f"  saved {OUT_DIR / 'orbit_MWPotential2014_samples.npz'}")
print("\nMWPotential2014 summary  (median +sig68 / -sig68):")
print_summary("No LMC",   peri_no_v2,  apo_no_v2,  e_no_v2)
print_summary("With LMC", peri_lmc_v2, apo_lmc_v2, e_lmc_v2)


# ===========================================================================
# 3. Gradient-MC (Fig 7 error bar)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 3: Gradient-MC, +/- 0.5 Gyr (McMillan17 + LMC), 1000 realisations")
print("=" * 70)


def _orbit_gradient_from_arr(ra_arr, dec_arr, vlos_arr, i0):
    if i0 < 1 or i0 > len(ra_arr) - 2:
        return np.nan, np.nan
    sd0 = np.sin(np.radians(dec_arr[:-1])); sd1 = np.sin(np.radians(dec_arr[1:]))
    cd0 = np.cos(np.radians(dec_arr[:-1])); cd1 = np.cos(np.radians(dec_arr[1:]))
    cdra = np.cos(np.radians(ra_arr[:-1] - ra_arr[1:]))
    seg = np.degrees(np.arccos(np.clip(sd0*sd1 + cd0*cd1*cdra, -1.0, 1.0)))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc = arc - arc[i0]
    if arc.min() > -1.0 or arc.max() < 1.0:
        return np.nan, np.nan
    v_p1 = float(np.interp(+1.0, arc, vlos_arr))
    v_m1 = float(np.interp(-1.0, arc, vlos_arr))
    slope = (v_p1 - v_m1) / 2.0
    dra_t  = (ra_arr[i0+1] - ra_arr[i0-1]) * np.cos(np.radians(dec_arr[i0]))
    ddec_t = (dec_arr[i0+1] - dec_arr[i0-1])
    pa = np.degrees(np.arctan2(dra_t, ddec_t)) % 360.0
    return slope * np.sin(np.radians(pa)), slope * np.cos(np.radians(pa))


def gradient_mc(pot, n_real, label, seed=2024,
                t_short_Gyr=0.5, n_step=N_STEP):
    print(f"\n-> {label}, {n_real} realisations, +/- {t_short_Gyr} Gyr ...")
    rng = np.random.default_rng(seed)
    p6_many = rng.normal(loc=PARAMS_6D, scale=ERRORS_6D, size=(n_real, 6))
    p6_many[:, 2] = np.clip(p6_many[:, 2], 0.1, None)
    ts_b = -np.linspace(0, t_short_Gyr, n_step) * u.Gyr
    ts_f =  np.linspace(0, t_short_Gyr, n_step) * u.Gyr
    orb_c_b = Orbit(PARAMS_6D, solarmotion=SOLAR_MOTION, radec=True, **get_physical(McMillan17))
    orb_c_f = Orbit(PARAMS_6D, solarmotion=SOLAR_MOTION, radec=True, **get_physical(McMillan17))
    orb_c_b.integrate(ts_b, pot); orb_c_f.integrate(ts_f, pot)
    ra_c   = np.concatenate([np.asarray(orb_c_b.ra(ts_b))[::-1],   np.asarray(orb_c_f.ra(ts_f))])
    dec_c  = np.concatenate([np.asarray(orb_c_b.dec(ts_b))[::-1],  np.asarray(orb_c_f.dec(ts_f))])
    vlos_c = np.concatenate([np.asarray(orb_c_b.vlos(ts_b))[::-1], np.asarray(orb_c_f.vlos(ts_f))])
    i0 = n_step - 1
    gE_c, gN_c = _orbit_gradient_from_arr(ra_c, dec_c, vlos_c, i0)

    orb_m_b = Orbit(p6_many, solarmotion=SOLAR_MOTION, radec=True, **get_physical(McMillan17))
    orb_m_f = Orbit(p6_many, solarmotion=SOLAR_MOTION, radec=True, **get_physical(McMillan17))
    orb_m_b.integrate(ts_b, pot); orb_m_f.integrate(ts_f, pot)
    ra_b = np.asarray(orb_m_b.ra(ts_b)); dec_b = np.asarray(orb_m_b.dec(ts_b)); v_b = np.asarray(orb_m_b.vlos(ts_b))
    ra_f = np.asarray(orb_m_f.ra(ts_f)); dec_f = np.asarray(orb_m_f.dec(ts_f)); v_f = np.asarray(orb_m_f.vlos(ts_f))
    gE = np.full(n_real, np.nan); gN = np.full(n_real, np.nan)
    for i in range(n_real):
        ra_i  = np.concatenate([ra_b[i, :][::-1],  ra_f[i, :]])
        dec_i = np.concatenate([dec_b[i, :][::-1], dec_f[i, :]])
        v_i   = np.concatenate([v_b[i, :][::-1],   v_f[i, :]])
        gE[i], gN[i] = _orbit_gradient_from_arr(ra_i, dec_i, v_i, i0)
    return dict(gE=gE, gN=gN, gE_c=gE_c, gN_c=gN_c, p6=p6_many)


GRAD_LMC = gradient_mc(pot_bwd_mcm, N_REAL, "McMillan17 + LMC (full 6D)")
gE_pct = np.nanpercentile(GRAD_LMC["gE"], [16, 50, 84])
gN_pct = np.nanpercentile(GRAD_LMC["gN"], [16, 50, 84])
print(f"  Central (no-error): gE = {GRAD_LMC['gE_c']:+.3f}, gN = {GRAD_LMC['gN_c']:+.3f}")
print(f"  gE 16/50/84:  {gE_pct[0]:+.3f}  {gE_pct[1]:+.3f}  {gE_pct[2]:+.3f}")
print(f"  gN 16/50/84:  {gN_pct[0]:+.3f}  {gN_pct[1]:+.3f}  {gN_pct[2]:+.3f}")
np.savez(OUT_DIR / "orbit_gradient_mc_1000_summary.npz",
         gE=GRAD_LMC["gE"], gN=GRAD_LMC["gN"],
         gE_c=GRAD_LMC["gE_c"], gN_c=GRAD_LMC["gN_c"],
         p6=GRAD_LMC["p6"], gE_pct=gE_pct, gN_pct=gN_pct)
print(f"  saved {OUT_DIR / 'orbit_gradient_mc_1000_summary.npz'}")


# ===========================================================================
# 4. Per-component MC (Fig A4 — 6 x 5 panel)
# ===========================================================================
print("\n" + "=" * 70)
print(f"STEP 4: Per-component MC, {N_PC} realisations x 6 components, +/- {T_GYR_PC} Gyr")
print("=" * 70)


def per_component_mc(pot, comp_key, n_real, seed,
                     t_short_Gyr=T_GYR_PC, n_step=N_STEP_PC):
    rng = np.random.default_rng(seed)
    err = np.zeros(6); err[COMP_IDX[comp_key]] = ERRORS_6D[COMP_IDX[comp_key]]
    p6_many = rng.normal(loc=PARAMS_6D, scale=err, size=(n_real, 6))
    if comp_key == "dist":
        p6_many[:, 2] = np.clip(p6_many[:, 2], 0.1, None)
    ts_b = -np.linspace(0, t_short_Gyr, n_step) * u.Gyr
    ts_f =  np.linspace(0, t_short_Gyr, n_step) * u.Gyr
    PHYS = get_physical(McMillan17); ro_, vo_ = PHYS["ro"], PHYS["vo"]
    orb_b = Orbit(p6_many, solarmotion=SOLAR_MOTION, radec=True, ro=ro_, vo=vo_)
    orb_f = Orbit(p6_many, solarmotion=SOLAR_MOTION, radec=True, ro=ro_, vo=vo_)
    orb_b.integrate(ts_b, pot); orb_f.integrate(ts_f, pot)
    def cat(b, f):
        b = np.asarray(b); f = np.asarray(f)
        return np.concatenate([b[:, ::-1], f[:, 1:]], axis=1)
    out = dict(ra=cat(orb_b.ra(ts_b),   orb_f.ra(ts_f)),
               dec=cat(orb_b.dec(ts_b), orb_f.dec(ts_f)),
               pmra=cat(orb_b.pmra(ts_b), orb_f.pmra(ts_f)),
               pmdec=cat(orb_b.pmdec(ts_b),orb_f.pmdec(ts_f)),
               vlos=cat(orb_b.vlos(ts_b), orb_f.vlos(ts_f)),
               dist=cat(orb_b.dist(ts_b), orb_f.dist(ts_f)),
               p6=p6_many)
    i0 = n_step - 1
    gE = np.full(n_real, np.nan); gN = np.full(n_real, np.nan)
    for i in range(n_real):
        gE[i], gN[i] = _orbit_gradient_from_arr(out["ra"][i, :],
                                                 out["dec"][i, :],
                                                 out["vlos"][i, :], i0)
    out["gE"] = gE; out["gN"] = gN
    return out


PC = {}
t0 = time.time()
for ki, key in enumerate(COMP_KEYS):
    if ERRORS_6D[COMP_IDX[key]] == 0:
        print(f"  {key:5s}: zero error -> skipped")
        continue
    PC[key] = per_component_mc(pot_bwd_mcm, key, N_PC, seed=1000+ki)
    print(f"  {key:5s} done ({time.time()-t0:.1f} s elapsed)")

# Fiducial trajectory
PHYS = get_physical(McMillan17); ro_, vo_ = PHYS["ro"], PHYS["vo"]
ts_b = -np.linspace(0, T_GYR_PC, N_STEP_PC) * u.Gyr
ts_f =  np.linspace(0, T_GYR_PC, N_STEP_PC) * u.Gyr
of_b = Orbit(PARAMS_6D, solarmotion=SOLAR_MOTION, radec=True, ro=ro_, vo=vo_)
of_f = Orbit(PARAMS_6D, solarmotion=SOLAR_MOTION, radec=True, ro=ro_, vo=vo_)
of_b.integrate(ts_b, pot_bwd_mcm); of_f.integrate(ts_f, pot_bwd_mcm)


def _cat(b, f):
    b = np.asarray(b); f = np.asarray(f)
    return np.concatenate([b[::-1], f[1:]])


fid = dict(ra=_cat(of_b.ra(ts_b),  of_f.ra(ts_f)),
           dec=_cat(of_b.dec(ts_b), of_f.dec(ts_f)),
           pmra=_cat(of_b.pmra(ts_b), of_f.pmra(ts_f)),
           pmdec=_cat(of_b.pmdec(ts_b),of_f.pmdec(ts_f)),
           vlos=_cat(of_b.vlos(ts_b), of_f.vlos(ts_f)),
           dist=_cat(of_b.dist(ts_b), of_f.dist(ts_f)))

# Save trajectories (for Figure A4)
save_kwargs = dict(comp_keys=np.array(COMP_KEYS),
                   params_6D=PARAMS_6D, errors_6D=ERRORS_6D,
                   n_pc=N_PC, n_step=N_STEP_PC, t_gyr=T_GYR_PC,
                   fid_ra=fid["ra"], fid_dec=fid["dec"],
                   fid_pmra=fid["pmra"], fid_pmdec=fid["pmdec"],
                   fid_vlos=fid["vlos"], fid_dist=fid["dist"])
for k in COMP_KEYS:
    if k not in PC: continue
    save_kwargs[f"{k}__ra"]    = PC[k]["ra"]
    save_kwargs[f"{k}__dec"]   = PC[k]["dec"]
    save_kwargs[f"{k}__pmra"]  = PC[k]["pmra"]
    save_kwargs[f"{k}__pmdec"] = PC[k]["pmdec"]
    save_kwargs[f"{k}__vlos"]  = PC[k]["vlos"]
    save_kwargs[f"{k}__dist"]  = PC[k]["dist"]
np.savez(OUT_DIR / "orbit_per_component_trajectories.npz", **save_kwargs)
print(f"  saved {OUT_DIR / 'orbit_per_component_trajectories.npz'}")

# Also save the gE/gN summary
sum_kwargs = dict(comp_keys=np.array(COMP_KEYS),
                  params_6D=PARAMS_6D, errors_6D=ERRORS_6D)
for k in COMP_KEYS:
    if k not in PC: continue
    sum_kwargs[f"gE_{k}"] = PC[k]["gE"]
    sum_kwargs[f"gN_{k}"] = PC[k]["gN"]
np.savez(OUT_DIR / "orbit_per_component_summary.npz", **sum_kwargs)
print(f"  saved {OUT_DIR / 'orbit_per_component_summary.npz'}")

print("\nPer-component sigmas (gE, gN):")
print(f"{'comp':<6s}  {'sigma_gE':>9s}  {'sigma_gN':>9s}  {'|sigma|':>8s}")
for key in COMP_KEYS:
    if key not in PC: continue
    sE = np.nanstd(PC[key]["gE"]); sN = np.nanstd(PC[key]["gN"])
    print(f"  {key:<4s}  {sE:>9.3f}  {sN:>9.3f}  {np.hypot(sE,sN):>8.3f}")

print("\nDone.  Notebook 01 can now read these npz files for plotting.")
