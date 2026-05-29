# Boötes III — Active Tidal Disruption with $S^5$ + Gaia DR3

This repository contains the code, intermediate data, figures, and LaTeX source for
**Li et al. (2026), "Boötes III: Active Tidal Disruption Confirmed with $S^5$ and Gaia DR3"**.
The paper itself is hosted separately on arXiv (and at the journal); this repository contains only the analysis pipeline and figures.

## Layout

```
boo3/
├── data/             ← raw input catalogues used by the notebooks
├── notebooks/        ← analysis notebooks + heavy-compute Python scripts
├── output/           ← intermediate products (GMM chains, orbit + spray caches; regenerable)
└── figures/          ← final paper figures (PDF)
```

## Reproducing the analysis

Each notebook is self-contained — all paths, constants, and modelling code live
inline so reading top-to-bottom gives the complete workflow. Top-level scripts
(`01_compute_gmm.py`, `01_compute_orbits.py`, `02_compute_spray.py`) handle the
heavy compute; the notebooks read their outputs and produce the paper figures.

The required Python environment is captured in `notebooks/requirements.txt`
(coming soon). Key packages: `astropy`, `emcee`, `galpy`, `agama` (for the N-body
ICs), `corner`, `matplotlib`, `tqdm`, `h5py`.

```sh
# Run from the repository root.
python notebooks/01_compute_gmm.py
python notebooks/01_compute_orbits.py
python notebooks/02_compute_spray.py all   # ~2–3 h
jupyter nbconvert --to notebook --execute --inplace notebooks/01_members_gmm_orbits.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_streamtrack_fig5.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_velocity_gradient_fig7_8.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_action_peri_sgr_fig9_10_11.ipynb
```

## Large input files

The Sgr stream catalogue of Vasiliev & Belokurov (2021) is too large for git
and will be hosted on Zenodo (link will be added when the paper is published):

| File | Size | Purpose | Source |
|---|---|---|---|
| `data/vasiliev2021_sgr_catalog.dat` | 7.7 MB | Sgr stream catalog | Vasiliev & Belokurov 2021 (Zenodo TBD) |

The $S^5$ DR2 catalogue used in this work is not yet public — it will be
released alongside the $S^5$ DR2 data paper (T. S. Li et al., in preparation,
expected summer 2026). The notebook scripts that consume `cat_s5_public_dr2.0_beta0.fits`
will be runnable once that release goes public; in the meantime they document
the queries and filters applied.

The N-body simulation snapshot used for the velocity-gradient comparison in
§6.2 of the paper is not redistributed with this repository; please contact the
authors for access.

## Citing this work

```bibtex
@ARTICLE{Li2026BooIII,
  author  = {{Li}, Ting S. and {S$^5$ Collaboration}},
  title   = "{Bo{\"o}tes III: Active Tidal Disruption Confirmed with $S^5$ and Gaia DR3}",
  journal = {ApJ},
  year    = 2026,
  note    = {in preparation}
}
```

## Compute provenance

The analysis pipeline and public-release structure were prepared with the help
of Anthropic's **Claude Code** (Claude Opus 4.7) as an interactive coding
assistant — for organising the analysis into the documented notebook + script
structure delivered here, regenerating auto-generated tables from the
notebooks, and running internal-consistency audits on the manuscript.  All
scientific results, design decisions, and conclusions are the responsibility of
the authors.

## License

Code under the MIT License (see `LICENSE`).  Paper text under CC BY 4.0.
