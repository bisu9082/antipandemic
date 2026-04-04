# Anti-Pandemic ML Surveillance — Code Repository

**Paper:** "Machine Learning Detection of Atypical Epidemiological Fingerprints for Early Warning of High-Consequence Outbreak Events: A 23-Year Retrospective Validation Using WHO Disease Outbreak Data"
**Target journal:** *The Lancet Digital Health*
**Repository:** https://github.com/bisu9082/antipandemic

---

## Overview

This repository contains the complete analysis code for a dual-purpose early-warning surveillance system that detects atypical epidemiological fingerprints of high-consequence outbreak events — applicable to both **pandemic early warning** and **CBRN biosurveillance** — using WHO Disease Outbreak News (DON) data spanning 1996–2019.

CDC Category A pathogens (Ebola, Marburg, Lassa, plague, anthrax, tularemia, botulism, smallpox) serve as the operational proxy for high-consequence events with pandemic escalation potential, based on their established PHEIC risk profiles and One Health frameworks.

**Key results:**
- M-FULL model AUC: **0·987 ± 0·004** (25 runs; 5-fold CV × 5 seeds)
- Quantitative-threshold baseline (B-0) AUC: **0·619 ± 0·019**
- Cohen's *d* = **5·31** (*p* = 4·3 × 10⁻¹⁹)
- Estimated advance warning: **8·7 days** over quantitative-threshold surveillance

---

## Repository Structure

```
antipandemic/
├── analysis_main.py      # Full ML pipeline: preprocessing → CV → statistics → SHAP
├── make_figures.py       # Publication figure generation (Fig 2–6)
├── requirements.txt      # Python package dependencies
├── data/
│   └── who_don_events.csv    # WHO DON processed dataset (place here)
└── outputs/
    ├── experiment_summary.json   # Final per-condition AUC statistics
    ├── figures/                  # Generated PNG figures
    └── tables/
        └── shap_importance.csv  # SHAP mean |values| per feature
```

---

## Installation

```bash
git clone https://github.com/bisu9082/antipandemic.git
cd antipandemic
pip install -r requirements.txt
```

Python 3.10 or 3.11 recommended.

---

## Usage

### 1. Run full ML pipeline

```bash
python analysis_main.py \
    --data    data/who_don_events.csv \
    --output  outputs/experiment_summary.json
```

This runs all 7 experimental conditions (175 CV runs total) and saves:
- `outputs/experiment_summary.json` — AUC statistics per condition
- `outputs/tables/shap_importance.csv` — SHAP feature importance

Add `--skip-cv` to skip cross-validation and compute SHAP only:

```bash
python analysis_main.py --data data/who_don_events.csv --skip-cv
```

### 2. Generate publication figures

```bash
python make_figures.py \
    --results outputs/experiment_summary.json \
    --shap    outputs/tables/shap_importance.csv \
    --outdir  outputs/figures
```

To regenerate only specific figures:

```bash
python make_figures.py --figs 2 6   # Only Fig2 and Fig6
```

**Figures produced:**

| File | Description |
|------|-------------|
| `Fig2_AUC_comparison.png` | Bar chart: AUC across 7 conditions |
| `Fig3_LeadTime.png` | Box plot: lead-time distributions |
| `Fig4_ROC.png` | ROC curves for B-0, B-2, M-FULL |
| `Fig5_Ablation.png` | Ablation: absolute AUC + incremental contribution |
| `Fig6_SHAP.png` | SHAP feature importance (M-FULL) |

---

## Experimental Conditions

| ID | Description |
|----|-------------|
| B-0 | Quantitative-threshold proxy: case count, death rate, calendar features only (no NLP, no anomaly detection) |
| B-1 | B-0 + Isolation Forest anomaly layer |
| B-2 | B-1 + CFR and notification lead-time features |
| M-IF | B-2 + Isolation Forest (anomaly layer) |
| M-RF | M-IF + Random Forest (stacking) |
| M-GCI | M-RF + Geographic Clustering Index (GCI) |
| M-FULL | M-GCI + Seasonal Mismatch Anomaly (SMA) + GCI×SMA interaction |

> **Note on B-0:** The quantitative-threshold baseline represents a traditional count-and-spread threshold approach. It is *not* intended to approximate NLP-based or media-scanning systems such as EIOS or HealthMap, which operate on fundamentally different signal types.

---

## Novel Features

**Geographic Clustering Index (GCI)**
Captures simultaneous multi-country reporting of the same pathogen within a ±30-day window:

```
GCI_{d,t} = 1 / N_{d,t}
```
where N_{d,t} is the number of countries reporting disease *d* within the temporal window around time *t*. Lower GCI → higher geographic clustering → higher threat signal.

**Seasonal Mismatch Anomaly (SMA)**
Quantifies deviation from 10-year historical seasonal baseline (Z-score, normalized [0,1]):

```
SMA_{d,t} = Φ( (cases_{d,t} - μ_{d,month}) / σ_{d,month} )
```

**GCI × SMA interaction term** captures events that are simultaneously geographically focused AND seasonally anomalous.

---

## Data

The WHO DON dataset used in this study is derived from publicly available WHO Disease Outbreak News (https://www.who.int/emergencies/disease-outbreak-news), covering 1996–2019 (*n* = 3,338 records). The 2020 data cutoff reflects WHO reporting disruption from COVID-19 pandemic response. The processed dataset (`who_don_events.csv`) is available in this repository.

**Note on `raw_results.csv`:** An early file (`outputs/tables/raw_results.csv`) reflects a preliminary pre-REFINE experimental run and does not match the final reported statistics. All final per-run AUC values are in `outputs/experiment_summary.json`.

---

## Citation

If you use this code, please cite:

```
[Citation will be added upon publication]
```

---

## License

MIT License — see LICENSE file for details.

---

## Contact

Correspondence: CBRN Defense Research Institute
GitHub issues: https://github.com/bisu9082/antipandemic/issues
