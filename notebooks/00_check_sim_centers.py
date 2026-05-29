"""Check the 6D centre of the N-body sim against the input fit.

Reports the median 6D in a small core aperture (default 0.05 deg ~ 3 arcmin)
for both the stars and the dark-matter components.  Expectation: the new sim,
which was integrated from the measured 6D backward, sits on the input 6D to
better than 0.001° on sky, 0.01 kpc in distance, and 0.1 km/s in vhel; minor
drift in vhel is expected at higher progenitor mass due to dynamical friction.

Run:
    python code/00_check_sim_centers.py
"""
from pathlib import Path
import h5py, numpy as np

PROJ        = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJ / "data"
SIM_FILE    = DATA_DIR / "boo3_v1_1e6.0Msun_McMillan.h5"
OUT_NPZ     = PROJ / "output" / "sim_centers.npz"

# --- Input 6D from the GMM fit (matches notebooks/01_compute_orbits.py) ---
TGT = dict(ra=209.5567, dec=26.5529, dist=48.5,
           pmra=-1.162, pmdec=-0.883, vhel=191.22)
TGT_ERR = dict(ra=0.3, dec=0.3, dist=1.8,
               pmra=0.017, pmdec=0.013, vhel=0.73)

CORE_DEG = 0.05      # 3 arcmin core aperture
KEYS = ("ra", "dec", "dist", "pmra", "pmdec", "vhel")


def wrap_ra(ra, ra0):
    """Wrap RA into (ra0 - 180, ra0 + 180]; continuous around ra0."""
    return ((ra - ra0 + 180) % 360) - 180 + ra0


def load(kind):
    """kind = 'star' or 'dm'"""
    with h5py.File(SIM_FILE, "r") as f:
        suf = "" if kind == "star" else "_dm"
        d = {k: f[f"{k}{suf}"][:] for k in KEYS}
    d["ra"] = wrap_ra(d["ra"], TGT["ra"])
    return d


def core_median(d, core_deg):
    cos_d = np.cos(np.radians(TGT["dec"]))
    r_sky = np.hypot((d["ra"] - TGT["ra"]) * cos_d, d["dec"] - TGT["dec"])
    m = r_sky < core_deg
    n = int(m.sum())
    if n < 30:
        return None, n
    med = {k: float(np.median(d[k][m])) for k in KEYS}
    return med, n


def main():
    print(f"Sim file     : {SIM_FILE.name}")
    print(f"Core aperture: {CORE_DEG*60:.1f} arcmin around "
          f"(RA, Dec) = ({TGT['ra']:.4f}, {TGT['dec']:.4f})")
    print(f"Target 6D    : "
          + ", ".join(f"{k}={TGT[k]:+.4f}±{TGT_ERR[k]:.3f}" for k in KEYS))
    print()
    head = f"{'kind':<5s}{'N':>7s}  " + "  ".join(f"{'d'+k:>9s}" for k in KEYS) + f"   {'in1σ':>5s}"
    print(head); print("-" * len(head))

    out = {}
    for kind in ("star", "dm"):
        d = load(kind)
        med, n = core_median(d, CORE_DEG)
        if med is None:
            print(f"{kind:<5s}{n:>7d}  [too few]"); continue
        diffs = {k: med[k] - TGT[k] for k in KEYS}
        in_1sig = sum(1 for k in KEYS if abs(diffs[k]) < TGT_ERR[k])
        dstr = "  ".join(f"{diffs[k]:+9.4f}" for k in KEYS)
        print(f"{kind:<5s}{n:>7d}  {dstr}   {in_1sig}/6")
        out[f"{kind}_med"]  = np.array([med[k]   for k in KEYS])
        out[f"{kind}_diff"] = np.array([diffs[k] for k in KEYS])
        out[f"{kind}_N"] = n

    out["keys"] = np.array(list(KEYS))
    out["target"] = np.array([TGT[k] for k in KEYS])
    out["target_err"] = np.array([TGT_ERR[k] for k in KEYS])
    out["core_deg"] = CORE_DEG
    OUT_NPZ.parent.mkdir(exist_ok=True)
    np.savez(OUT_NPZ, **out)
    print(f"\nSaved {OUT_NPZ}")


if __name__ == "__main__":
    main()
