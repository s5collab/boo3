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

The N-body simulation snapshot (`data/boo3_v1_1e6.0Msun_McMillan.h5`, ~92 MB)
used for the velocity-gradient comparison in §6.2 of the paper is not included
in this repository due to its size; please contact the authors for access.

## Key data products released here

Two data products from the paper are distributed in this repository so that
the membership analysis can be inspected and reproduced without access to the
non-public S$^5$ DR2 raw catalogue:

| File | Contents |
|---|---|
| **`output/boo3_21mem_machine_readable.csv`** | The 21 high-probability Boötes III RGB members from the GMM fit ($P_{\rm mem} > 0.95$) — the same stars listed in Table 3 of the paper. Columns: Gaia DR3 `source_id`, sky position (`ra`, `dec`, plus tangent-plane offsets from the new centroid), dereddened DECam $g_0, r_0$, $S^5$ heliocentric RV and error, $S^5$ [Fe/H] and error, Gaia DR3 proper motions and errors with the correlation coefficient, $S^5$ S/N and the GMM-input quality flag, and the median GMM membership probability with its 16/84-percentile interval. This is the machine-readable companion intended for AAS submission. |
| **`data/boo3_input_120.csv`** | The 120-star input catalog to which the Gaussian mixture model is fit — the small post-cut "stamp" of $S^5$ DR2 around Boötes III that survives our quality + ellipse + PM-box cuts. Although $S^5$ DR2 itself is not yet public (release expected summer 2026, T. S. Li et al., in preparation), distributing this stamp lets anyone re-run the GMM and reproduce the 21-member list using only the public code in this repository. |

To reproduce the membership analysis end-to-end:

```sh
python notebooks/01_compute_gmm.py
```

The script automatically loads `data/boo3_input_120.csv` if it is present and
skips the (S$^5$-DR2-dependent) build-from-scratch step.  The 21-member list
written to `output/boo3_21mem_machine_readable.csv` is then produced by the
companion notebook `notebooks/01_members_gmm_orbits.ipynb` (cell 8), alongside
the LaTeX Table 3 in the paper.

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
