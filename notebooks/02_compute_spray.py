"""
02_compute_spray.py — fardal15 spray sampling for notebook 02 (Figure 5).

Run AFTER `01_compute_gmm.py` (which writes the new 6D into
output/boo3_gmm_samples.h5).  Run BEFORE `02_streamtrack_fig5.ipynb`.

Usage
-----
  # default: just the fiducial spray (~13 min; for quick sanity-check)
  python 02_compute_spray.py
  # explicit single variation:
  python 02_compute_spray.py fid
  # full set (overnight; ~2-2.5 h on this machine):
  python 02_compute_spray.py all
  # custom subset:
  python 02_compute_spray.py mw05 mw2 lmc2

Each run produces (one per variation):
  output/spray_cache_new6D/spray_<key>.pkl
containing the particles (RA, Dec, PM, vlos, dist), the raw (xv, dt) for
streamTrack, and provenance (mw_factor, lmc_factor, solar_motion, ...).

The 9-variation table follows the paper's Fig 5 grid.
"""
import os, sys, time, pickle, warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u

from galpy.orbit import Orbit
from galpy.df import fardal15spraydf
from galpy.util import galpyWarning
from galpy.util.conversion import get_physical
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce, HernquistPotential,
    MovingObjectPotential, NonInertialFrameForce,
    evaluateRforces, evaluatephitorques, evaluatezforces,
)
from galpy.potential.mwpotentials import McMillan17 as _McMillan17_base

warnings.filterwarnings("ignore", category=galpyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Paths ----
NB_DIR    = Path(__file__).resolve().parent
PROJ      = NB_DIR.parent
OUT_DIR   = PROJ / "output"
CACHE_DIR = OUT_DIR / "spray_cache_new6D"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GMM_H5    = OUT_DIR / "boo3_gmm_samples.h5"

# ---- NEW Boo III 6D (kept in sync with 01_compute_gmm.py + 01_compute_orbits.py) ----
RA_BOO3, DEC_BOO3 = 209.5567, 26.5529
DIST_KPC          = 48.47   # 5-RRL Garofalo+22 mean; sprays in spray_cache_new6D regenerated at this value 2026-05-29.
RHALF_CIRC_AM     = 33.03
RHALF_CIRC_KPC    = DIST_KPC * np.tan(np.radians(RHALF_CIRC_AM/60.0))
SOLAR_MOTION_FID  = [-12.9, 12.5, 7.78]

# ---- Run knobs (same as the original notebook 02) ----
N_PER_ARM        = 1000
TDISRUPT         = 3.0      # Gyr
N_STEP           = 10001
TRACK_N_DENSE    = 1001
TRACK_TIME_RANGE = 3.0
SMOOTHING_FACTOR = 4.0

# ---- Load GMM medians ----
samp = pd.read_hdf(GMM_H5, key="samples")
vhel_med  = float(samp["vhel_0"].median())
pmra_med  = float(samp["pmr_0"].median())
pmdec_med = float(samp["pmd_0"].median())
sig_v_med = float(np.median(10 ** samp["log_sig_vhel"].values))

G_KPC_KMS_MSUN = 4.30091e-6
PROG_MASS = 4.0 * sig_v_med**2 * RHALF_CIRC_KPC / G_KPC_KMS_MSUN
PARAMS_6D = [RA_BOO3, DEC_BOO3, DIST_KPC, pmra_med, pmdec_med, vhel_med]

print("=" * 70)
print("02_compute_spray.py — NEW 6D (this work)")
print("=" * 70)
print(f"  centre  : ({RA_BOO3:.4f}, {DEC_BOO3:.4f}), d = {DIST_KPC} kpc")
print(f"  v_hel   : {vhel_med:+.3f} km/s")
print(f"  pmra    : {pmra_med:+.4f} mas/yr")
print(f"  pmdec   : {pmdec_med:+.4f} mas/yr")
print(f"  sigma_v : {sig_v_med:.3f} km/s")
print(f"  R_h(circ) = {RHALF_CIRC_KPC*1000:.1f} pc")
print(f"  PROG_MASS = {PROG_MASS:.3e} Msun  (King: 4 sigma_v^2 R_h / G)")


# ---- Variation table (mirror nb02 cell 10) ----
VARIATIONS = [
    # key,  label,             color,             mw_factor, lmc_factor, solar_motion,             use_lmc
    ("fid",   "Fiducial",        "deepskyblue",     1.0,  1.0,  SOLAR_MOTION_FID,           True),
    ("lmc2",  "LMC mass x2",     "salmon",          1.0,  2.0,  SOLAR_MOTION_FID,           True),
    ("lmc05", "LMC mass x0.5",   "pink",            1.0,  0.5,  SOLAR_MOTION_FID,           True),
    ("mw15",  "MW mass x1.5",    "orchid",          1.5,  1.0,  SOLAR_MOTION_FID,           True),
    ("mw2",   "MW mass x2",      "purple",          2.0,  1.0,  SOLAR_MOTION_FID,           True),
    ("mw05",  "MW mass x0.5",    "magenta",         0.5,  1.0,  SOLAR_MOTION_FID,           True),
    ("vp233", "Vphi = 233",      "gold",            1.0,  1.0,  [-12.9,  0.0,  7.78],       True),
    ("vp258", "Vphi = 258",      "brown",           1.0,  1.0,  [-12.9, 25.0,  7.78],       True),
    ("nolmc", "No LMC",          "cornflowerblue",  1.0,  1.0,  SOLAR_MOTION_FID,           False),
]
VARIATION_BY_KEY = {v[0]: v for v in VARIATIONS}
ALL_KEYS = [v[0] for v in VARIATIONS]


def _spray_cache_path(key):
    return CACHE_DIR / f"spray_{key}.pkl"


def build_lmc_potential(mw_factor, lmc_factor, solar_motion, use_lmc):
    base = [p for p in _McMillan17_base]
    base[1] = base[1] * mw_factor
    if not use_lmc:
        return base, base

    mass_lmc = 1.38e11 * lmc_factor
    rscale, rhm = 16.09, 16.09 * (1 + np.sqrt(2))
    orb_lmc = Orbit.from_name("LMC", solarmotion=solar_motion,
                                **get_physical(base))
    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=mass_lmc * u.Msun, rhm=rhm * u.kpc, dens=base[1],
        **get_physical(base))
    ts = np.linspace(0, 5, 1001) * u.Gyr
    orb_lmc.integrate(-ts, base + [cdf])

    lmcpot = HernquistPotential(amp=2 * mass_lmc * u.Msun, a=rscale * u.kpc,
                                 **get_physical(base))
    moving = MovingObjectPotential(orb_lmc, pot=lmcpot, **get_physical(base))
    loc = 1e-4
    af = lambda t: evaluateRforces(moving, loc, 0., phi=0., t=t, use_physical=False)
    bf = lambda t: evaluatephitorques(moving, loc, 0., phi=0., t=t, use_physical=False) / loc
    cf = lambda t: evaluatezforces(moving, loc, 0., phi=0., t=t, use_physical=False)
    ti = orb_lmc.time(use_physical=False)[::-1]
    aa = np.array([af(t) for t in ti])
    ab = np.array([bf(t) for t in ti])
    ac = np.array([cf(t) for t in ti])
    nip = NonInertialFrameForce(a0=[
        lambda t: np.interp(t, ti, aa),
        lambda t: np.interp(t, ti, ab),
        lambda t: np.interp(t, ti, ac),
    ])
    return base + [nip, moving], base


def sample_one_variation(key):
    pkl = _spray_cache_path(key)
    if pkl.exists():
        return key, 0.0, "cached"

    np.random.seed(abs(hash(key)) % (2**32 - 1))
    _, label, color, mw_f, lmc_f, sm, use_lmc = VARIATION_BY_KEY[key]
    pot, rtpot = build_lmc_potential(mw_f, lmc_f, sm, use_lmc)
    phys = get_physical(_McMillan17_base)
    ro, vo = phys["ro"], phys["vo"]
    orb_boo3 = Orbit(PARAMS_6D, radec=True, solarmotion=sm, ro=ro, vo=vo)

    spdf = fardal15spraydf(progenitor_mass=PROG_MASS * u.Msun,
                           progenitor=orb_boo3, pot=pot,
                           tdisrupt=TDISRUPT * u.Gyr, rtpot=rtpot,
                           tail="both", ro=ro, vo=vo)

    t0 = time.time()
    xv, dt = spdf.sample(n=2 * N_PER_ARM, return_orbit=False, returndt=True,
                         integrate=True)
    orb_samp = Orbit(np.column_stack([xv[0], xv[1], xv[2], xv[3], xv[4], xv[5]]),
                     ro=ro, vo=vo, solarmotion=sm)
    particles = dict(
        ra=np.asarray(orb_samp.ra()),     dec=np.asarray(orb_samp.dec()),
        pmra=np.asarray(orb_samp.pmra()), pmdec=np.asarray(orb_samp.pmdec()),
        vlos=np.asarray(orb_samp.vlos()), dist=np.asarray(orb_samp.dist()),
        n_lead=N_PER_ARM,
    )
    cache = dict(
        xv=np.asarray(xv), dt=np.asarray(dt), particles=particles,
        key=key, label=label, color=color, mw_factor=mw_f, lmc_factor=lmc_f,
        solar_motion=tuple(sm), use_lmc=use_lmc,
        prog_mass=PROG_MASS, tdisrupt=TDISRUPT, n_per_arm=N_PER_ARM,
        n_step=N_STEP, sigma_v_used=sig_v_med,
        rh_kpc=RHALF_CIRC_KPC, dist_kpc=DIST_KPC,
        params_6D=PARAMS_6D,
    )
    with open(pkl, "wb") as f:
        pickle.dump(cache, f)
    return key, time.time() - t0, "fresh"


def main():
    args = sys.argv[1:]
    if not args:
        keys = ["fid"]
    elif args == ["all"]:
        keys = ALL_KEYS
    else:
        bad = [a for a in args if a not in VARIATION_BY_KEY]
        if bad:
            print(f"unknown variation key(s): {bad}\nvalid: {ALL_KEYS}")
            sys.exit(1)
        keys = args

    print(f"\nVariations to spray: {keys}")
    print(f"Cache dir: {CACHE_DIR}\n")
    t_overall = time.time()
    results = []
    for k in keys:
        print(f">>> {k}", flush=True)
        res = sample_one_variation(k)
        results.append(res)
        print(f"    -> {res[2]}, {res[1]:.1f}s\n", flush=True)
    print(f"Wall time = {(time.time() - t_overall)/60:.2f} min")
    print()
    print(f"{'key':<8} {'time(s)':>9}  status")
    for k, t, status in results:
        print(f"{k:<8} {t:>9.1f}  {status}")


if __name__ == "__main__":
    main()
