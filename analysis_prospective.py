"""
analysis_prospective.py — Supplementary Validation Experiments
================================================================
Project  : 안티펜데믹 (Anti-Pandemic)
Paper    : "Machine Learning Detection of Atypical Epidemiological Fingerprints
           for Early Warning of High-Consequence Outbreak Events"
Journal  : The Lancet Digital Health (under review)
Author   : Ku Kang (cbr6290@mnd.go.kr)

Three experiments that strengthen the primary analysis_main.py results:

  Exp 1 — Pseudo-prospective temporal holdout
           Train 1996–2014  →  Test 2015–2019 (strict temporal separation)
           Addresses: "retrospective design cannot demonstrate temporal generalisation"

  Exp 2 — Feature integrity ablation
           M-FULL minus CFR  |  M-FULL minus lead_time  |  M-FULL minus both
           Addresses: "CFR is definitionally correlated with the label" and
                      "lead_time_days is a leaky feature"

  Exp 3 — COVID-19 early-detection case study
           Apply 1996–2019-trained M-FULL to reconstructed early WHO DON
           reports for SARS-CoV-2 (Dec 2019 – Mar 2020)
           Addresses: "no external validation on data outside training period"

Usage
-----
    python analysis_prospective.py \
        --data    data/who_don_events.csv \
        --output  outputs/prospective/

Outputs
-------
    prospective_summary.json      — all experiment results
    temporal_holdout_roc.png      — ROC curves, temporal holdout
    feature_integrity_bar.png     — AUC bar chart, feature ablation
    covid19_probability_trace.png — COVID-19 detection probability over time
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# Import shared utilities from analysis_main
from analysis_main import (
    load_and_preprocess,
    add_novel_features,
    compute_gci,
    compute_sma,
    FEAT_FULL,
    CDC_A_PATHOGENS,
    SEEDS,
    N_SPLITS,
    BOOTSTRAP_N,
)

warnings.filterwarnings('ignore')

# ── Colour palette (Lancet-compatible) ─────────────────────────────────────────
PALETTE = {
    'navy':  '#1A3F6F',
    'red':   '#C0392B',
    'grey':  '#6B6B6B',
    'green': '#27AE60',
    'amber': '#E67E22',
    'light': '#AED6F1',
}

FEAT_NO_CFR = [f for f in FEAT_FULL if f != 'death_rate']
FEAT_NO_LT  = [f for f in FEAT_FULL if f != 'lead_time_days']
FEAT_NO_SUSPECT = [f for f in FEAT_FULL if f not in ('death_rate', 'lead_time_days')]

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — PSEUDO-PROSPECTIVE TEMPORAL HOLDOUT
# ══════════════════════════════════════════════════════════════════════════════

def run_temporal_holdout(df: pd.DataFrame) -> dict:
    """
    Strict temporal split: train 1996–2014, evaluate 2015–2019.

    Rationale for 2014 cutoff:
      - The 2014–2016 West Africa Ebola epidemic began in December 2013.
        Cutting at end-2014 ensures the model has seen early-phase Ebola
        but predicts the epidemic's continuation and all subsequent events
        on truly unseen data.
      - This mirrors the operational scenario where a model trained on
        historical data is deployed to screen future notifications.

    Returns
    -------
    dict with keys: train_n, test_n, conditions (per-condition AUC)
    """
    train = df[df['Year'] <= 2014].copy()
    test  = df[df['Year'] >= 2015].copy()

    print(f"  Temporal split — Train: n={len(train):,} (1996–2014) | "
          f"Test: n={len(test):,} (2015–2019)")
    print(f"  Train prevalence: {train['label'].mean():.1%}  |  "
          f"Test prevalence: {test['label'].mean():.1%}")

    results = {
        'train_n': len(train),
        'test_n':  len(test),
        'train_years': '1996–2014',
        'test_years':  '2015–2019',
        'train_prevalence': float(train['label'].mean()),
        'test_prevalence':  float(test['label'].mean()),
        'conditions': {},
    }

    conditions = {
        'B-0': {
            'feats': ['log_cases', 'death_rate', 'Month', 'DayOfYear'],
            'rf_n': 100, 'use_if': False,
        },
        'M-FULL': {
            'feats': FEAT_FULL, 'rf_n': 500, 'use_if': True,
        },
        'M-noSuspect': {
            'feats': FEAT_NO_SUSPECT, 'rf_n': 500, 'use_if': True,
        },
    }

    all_roc = {}
    for cond_name, cfg in conditions.items():
        feats = cfg['feats']
        # Align feature columns — fill missing with 0
        for f in feats:
            if f not in train.columns:
                train[f] = 0.0
            if f not in test.columns:
                test[f] = 0.0

        X_tr = train[feats].fillna(0).values
        y_tr = train['label'].values
        X_te = test[feats].fillna(0).values
        y_te = test['label'].values

        auc_runs = []
        fpr_list, tpr_list = [], []

        for seed in SEEDS:
            if cfg.get('use_if'):
                iforest = IsolationForest(
                    n_estimators=200, contamination=0.18, random_state=seed
                )
                iforest.fit(X_tr)
                if_tr = (-iforest.score_samples(X_tr)).reshape(-1, 1)
                if_te = (-iforest.score_samples(X_te)).reshape(-1, 1)
                X_tr2 = np.hstack([X_tr, if_tr])
                X_te2 = np.hstack([X_te, if_te])
            else:
                X_tr2, X_te2 = X_tr, X_te

            rf = RandomForestClassifier(
                n_estimators=cfg['rf_n'],
                class_weight='balanced',
                random_state=seed, n_jobs=1
            )
            rf.fit(X_tr2, y_tr)
            proba = rf.predict_proba(X_te2)[:, 1]
            auc_val = roc_auc_score(y_te, proba)
            auc_runs.append(auc_val)

            fpr, tpr, _ = roc_curve(y_te, proba)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        auc_mean = float(np.mean(auc_runs))
        auc_std  = float(np.std(auc_runs, ddof=1))
        print(f"    {cond_name:15s}  AUC = {auc_mean:.4f} ± {auc_std:.4f}")

        results['conditions'][cond_name] = {
            'auc_mean': round(auc_mean, 4),
            'auc_std':  round(auc_std, 4),
            'n_runs':   len(auc_runs),
        }
        all_roc[cond_name] = {'fpr': fpr_list, 'tpr': tpr_list,
                              'auc_mean': auc_mean}

    return results, all_roc


def plot_temporal_roc(all_roc: dict, out_dir: Path) -> None:
    """ROC curves for temporal holdout (2015–2019 test set)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    styles = {
        'B-0':        ('--', PALETTE['grey'],  'B-0 (Quantitative-threshold)'),
        'M-FULL':     ('-',  PALETTE['navy'],  'M-FULL (all features)'),
        'M-noSuspect': ('-.', PALETTE['green'], 'M-FULL (GCI/SMA only — no CFR/LT)'),
    }

    for cond, (ls, col, label) in styles.items():
        if cond not in all_roc:
            continue
        roc_data = all_roc[cond]
        # Interpolate to common FPR grid
        fpr_grid = np.linspace(0, 1, 200)
        tpr_interp = np.array([
            np.interp(fpr_grid, fpr, tpr)
            for fpr, tpr in zip(roc_data['fpr'], roc_data['tpr'])
        ])
        tpr_mean = tpr_interp.mean(axis=0)
        tpr_std  = tpr_interp.std(axis=0)
        ax.plot(fpr_grid, tpr_mean, ls=ls, color=col, lw=2,
                label=f"{label} (AUC={roc_data['auc_mean']:.3f})")
        ax.fill_between(fpr_grid, tpr_mean - tpr_std, tpr_mean + tpr_std,
                        alpha=0.12, color=col)

    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC — Pseudo-prospective Holdout\n(Train 1996–2014; Test 2015–2019)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(out_dir / 'temporal_holdout_roc.png', dpi=300)
    plt.close(fig)
    print("  ✓ temporal_holdout_roc.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — FEATURE INTEGRITY ABLATION
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_integrity(df: pd.DataFrame) -> dict:
    """
    5-fold CV × 5 seeds for three feature-removal conditions:
      M-noCFR     : M-FULL without death_rate (Case Fatality Rate)
      M-noLT      : M-FULL without lead_time_days (Notification Lead Time)
      M-noSuspect : M-FULL without both CFR and lead_time

    These ablations are designed to isolate the contribution of GCI and SMA
    under the most conservative assumptions — i.e., assuming CFR and
    lead_time encode definitional or leaky information.

    If AUC remains substantially above B-0 (~0.619) even without CFR and
    lead_time, this demonstrates that the spatial (GCI) and temporal (SMA)
    features carry genuine, independent discriminative signal.
    """
    feat_conditions = {
        'M-FULL':      FEAT_FULL,
        'M-noCFR':     FEAT_NO_CFR,
        'M-noLT':      FEAT_NO_LT,
        'M-noSuspect': FEAT_NO_SUSPECT,
        'B-0':         ['log_cases', 'death_rate', 'Month', 'DayOfYear'],
    }

    results = {}
    X_all = {name: df[feats].fillna(0).values
             for name, feats in feat_conditions.items()}
    y = df['label'].values

    for cond_name, feats in feat_conditions.items():
        X = X_all[cond_name]
        auc_runs = []
        use_if = cond_name != 'B-0'
        rf_n   = 100 if cond_name == 'B-0' else 500

        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=seed)
            for tr_idx, te_idx in skf.split(X, y):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                if use_if:
                    iforest = IsolationForest(
                        n_estimators=200, contamination=0.18, random_state=seed
                    )
                    iforest.fit(X_tr)
                    if_tr = (-iforest.score_samples(X_tr)).reshape(-1, 1)
                    if_te = (-iforest.score_samples(X_te)).reshape(-1, 1)
                    X_tr2 = np.hstack([X_tr, if_tr])
                    X_te2 = np.hstack([X_te, if_te])
                else:
                    X_tr2, X_te2 = X_tr, X_te

                rf = RandomForestClassifier(
                    n_estimators=rf_n,
                    class_weight='balanced',
                    random_state=seed, n_jobs=1
                )
                rf.fit(X_tr2, y_tr)
                proba = rf.predict_proba(X_te2)[:, 1]
                auc_runs.append(roc_auc_score(y_te, proba))

        mean_ = float(np.mean(auc_runs))
        std_  = float(np.std(auc_runs, ddof=1))
        n_removed = len(FEAT_FULL) - len(feat_conditions[cond_name])
        print(f"    {cond_name:15s}  AUC = {mean_:.4f} ± {std_:.4f}  "
              f"(features removed vs M-FULL: {n_removed})")

        results[cond_name] = {
            'auc_mean': round(mean_, 4),
            'auc_std':  round(std_, 4),
            'n_features': len(feat_conditions[cond_name]),
            'features_removed': [
                f for f in FEAT_FULL if f not in feat_conditions[cond_name]
            ],
        }

    return results


def plot_feature_integrity(results: dict, out_dir: Path) -> None:
    """Bar chart comparing feature-integrity ablation AUC values."""
    order = ['B-0', 'M-noSuspect', 'M-noCFR', 'M-noLT', 'M-FULL']
    labels = [
        'B-0\n(Quant-Threshold)',
        'M-noSuspect\n(GCI/SMA only)',
        'M-noCFR\n(no CFR)',
        'M-noLT\n(no Lead Time)',
        'M-FULL\n(all features)',
    ]
    means = [results[c]['auc_mean'] for c in order]
    stds  = [results[c]['auc_std']  for c in order]
    colors = [
        PALETTE['grey'], PALETTE['green'],
        PALETTE['amber'], PALETTE['red'], PALETTE['navy'],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor='white', linewidth=0.8,
                  error_kw=dict(elinewidth=1.2, ecolor='#555555'))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.006,
                f"{m:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.5, 1.07)
    ax.set_ylabel('AUC-ROC (mean ± SD, 25 runs)')
    ax.set_title('Feature Integrity Ablation\n(5-fold CV × 5 seeds)',
                 fontsize=10)
    ax.axhline(0.5, color='#AAAAAA', lw=0.8, ls='--', label='Chance (0.5)')
    ax.axhline(results['B-0']['auc_mean'], color=PALETTE['grey'],
               lw=1.0, ls=':', alpha=0.7, label='B-0 level')

    # Annotate the GCI/SMA-only model with a star
    gs_idx = order.index('M-noSuspect')
    ax.annotate('GCI + SMA\nalone',
                xy=(gs_idx, means[gs_idx] + 0.012),
                xytext=(gs_idx + 0.6, means[gs_idx] + 0.04),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=8, ha='left')

    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_dir / 'feature_integrity_bar.png', dpi=300)
    plt.close(fig)
    print("  ✓ feature_integrity_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — COVID-19 EARLY DETECTION CASE STUDY
# ══════════════════════════════════════════════════════════════════════════════

def run_covid19_case_study(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Apply M-FULL trained on 1996–2019 to reconstructed early SARS-CoV-2
    WHO Disease Outbreak News reports (December 2019 – March 2020).

    Data reconstruction rationale:
    ─────────────────────────────
    WHO issued its first Disease Outbreak News entry for "Pneumonia of unknown
    cause — China" on 5 January 2020, describing 44 cases (no deaths) with
    onset dates from 8 December 2019. Subsequent WHO DON entries tracked
    the outbreak through early 2020.

    We reconstruct nine key reporting milestones from WHO and published
    literature (WHO DON archives; Huang et al., Lancet 2020;
    Wu et al., NEJM 2020), extracting the feature values that would have
    been available to the M-FULL model at each reporting date:

      - log_cases, death_rate     : from cumulative case/death counts
      - GCI                       : = 1.0 initially (China only),
                                    declining as international spread
      - SMA                       : high (novel pathogen with no seasonal
                                    baseline → treated as maximum anomaly)
      - lead_time_days            : estimated from Dec 8 onset to report date
      - Month, DayOfYear, Year    : from report date

    The M-FULL model was re-trained on the full 1996–2019 cohort (all 5
    seeds, then ensembled) and its probability score applied to each
    reconstructed feature vector.

    Key question: at what point would M-FULL have first flagged the
    COVID-19 cluster at P ≥ 0.50, relative to WHO's formal PHEIC
    declaration (30 January 2020)?
    """
    print("  Training M-FULL on full 1996–2019 cohort for COVID-19 case study…")

    # ── Train ensemble on full dataset ──
    feats = FEAT_FULL
    X_all = df[feats].fillna(0).values
    y_all = df['label'].values

    ensemble_models = []
    for seed in SEEDS:
        iforest = IsolationForest(
            n_estimators=200, contamination=0.18, random_state=seed
        )
        iforest.fit(X_all)
        if_scores = (-iforest.score_samples(X_all)).reshape(-1, 1)
        X_full = np.hstack([X_all, if_scores])

        rf = RandomForestClassifier(
            n_estimators=500, class_weight='balanced',
            random_state=seed, n_jobs=1
        )
        rf.fit(X_full, y_all)
        ensemble_models.append((iforest, rf))

    # ── Reconstruct early COVID-19 WHO DON reports ──
    # Columns: date_label, log_cases, death_rate, lead_time, Month, DayOfYear,
    #          Year, GCI, SMA, GCI_SMA
    # Sources:
    #  WHO DON 05 Jan 2020: 44 cases, 0 deaths (onset Dec 8)
    #  WHO DON 12 Jan 2020: 41 cases confirmed, 0 deaths; GCI still 1.0
    #  WHO DON 17 Jan 2020: 62 cases, 2 deaths; first case Thailand
    #  WHO DON 22 Jan 2020: 314 cases, 6 deaths; China, Thailand, Japan, Korea
    #  WHO PHEIC 30 Jan 2020: 7818 cases, 170 deaths; 18 countries
    #  WHO DON 07 Feb 2020: 31161 cases, 636 deaths; 25 countries
    #  WHO DON 29 Feb 2020: 85403 cases, 2924 deaths; 53 countries
    #  WHO Pandemic 11 Mar 2020: 118,000+ cases, 4291 deaths; 114 countries

    reports = [
        # date_label,    cases, deaths, n_countries, days_since_onset
        ('2020-01-05',    44,    0,    1,   28),   # First WHO DON
        ('2020-01-12',    41,    1,    1,   35),   # Confirmed cases
        ('2020-01-17',    62,    2,    2,   40),   # First int'l case (Thailand)
        ('2020-01-22',   314,    6,    4,   45),   # Rapid escalation
        ('2020-01-30',  7818,  170,   18,   53),   # PHEIC declared
        ('2020-02-07', 31161,  636,   25,   61),   # 4 weeks post-first DON
        ('2020-02-29', 85403, 2924,   53,   83),   # Late February
        ('2020-03-11',118000, 4291,  114,   94),   # Pandemic declared
    ]

    # SMA for a novel pathogen with no WHO DON history:
    # The pathogen 'pneumonia of unknown cause' has no prior seasonal baseline.
    # SMA is computed as maximum anomaly (clipped SMA = 1.0) because the
    # z-score is undefined and defaults to +∞ → clipped to upper bound.
    # This is the correct behaviour: genuinely novel events have no baseline
    # to compare against, so the model treats them as maximally anomalous.
    SMA_NOVEL = 1.0  # upper bound of normalised SMA

    covid_features = []
    for date_label, cases, deaths, n_countries, lt_days in reports:
        dt = pd.to_datetime(date_label)
        log_c  = float(np.log1p(cases))
        cfr    = float(deaths / max(cases, 1))
        gci    = float(1.0 / max(n_countries, 1))
        sma    = SMA_NOVEL  # novel pathogen — undefined baseline → max anomaly
        gcisma = gci * sma
        covid_features.append({
            'date':           date_label,
            'cases':          cases,
            'deaths':         deaths,
            'n_countries':    n_countries,
            'log_cases':      log_c,
            'death_rate':     cfr,
            'lead_time_days': float(lt_days),
            'Month':          float(dt.month),
            'DayOfYear':      float(dt.dayofyear),
            'Year':           float(dt.year),
            'GCI':            gci,
            'SMA':            sma,
            'GCI_SMA':        gcisma,
        })

    covid_df = pd.DataFrame(covid_features)
    X_covid = covid_df[feats].values

    # ── Score with ensemble ──
    probas = []
    for iforest, rf in ensemble_models:
        if_score = (-iforest.score_samples(X_covid)).reshape(-1, 1)
        X_c = np.hstack([X_covid, if_score])
        probas.append(rf.predict_proba(X_c)[:, 1])

    covid_df['proba_mean'] = np.mean(probas, axis=0)
    covid_df['proba_std']  = np.std(probas,  axis=0)

    print("  COVID-19 early detection results:")
    for _, row in covid_df.iterrows():
        flag = '★ FLAGGED' if row['proba_mean'] >= 0.50 else '  below threshold'
        print(f"    {row['date']}  cases={row['cases']:>6,}  "
              f"countries={row['n_countries']:>3}  "
              f"P(high-risk)={row['proba_mean']:.3f} ± {row['proba_std']:.3f}"
              f"  {flag}")

    # ── Identify first detection date ──
    flagged = covid_df[covid_df['proba_mean'] >= 0.50]
    first_flag = flagged.iloc[0]['date'] if len(flagged) > 0 else 'Never'
    pheic_date = '2020-01-30'
    if first_flag != 'Never':
        days_before_pheic = (
            pd.to_datetime(pheic_date) - pd.to_datetime(first_flag)
        ).days
        print(f"\n  First detection: {first_flag}  "
              f"({days_before_pheic} days before PHEIC declaration)")
    else:
        days_before_pheic = None
        print("  Model did not exceed threshold in this time window.")

    return {
        'first_flag_date': first_flag,
        'days_before_pheic': days_before_pheic,
        'pheic_date': pheic_date,
        'pandemic_date': '2020-03-11',
        'report_trajectory': covid_df[[
            'date', 'cases', 'deaths', 'n_countries',
            'GCI', 'SMA', 'proba_mean', 'proba_std'
        ]].to_dict('records'),
    }, covid_df


def plot_covid_trace(covid_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Two-panel figure:
    (a) M-FULL probability score over time (with ±1 SD band)
    (b) GCI and SMA component values over the same period
    """
    dates = pd.to_datetime(covid_df['date'])
    date_labels = [d.strftime('%b %d') for d in dates]
    x = np.arange(len(dates))

    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Panel (a): probability trace ──
    ax_a = fig.add_subplot(gs[0])
    ax_a.fill_between(
        x,
        covid_df['proba_mean'] - covid_df['proba_std'],
        covid_df['proba_mean'] + covid_df['proba_std'],
        alpha=0.2, color=PALETTE['navy'], label='±1 SD (5 seeds)'
    )
    ax_a.plot(x, covid_df['proba_mean'], 'o-',
              color=PALETTE['navy'], lw=2, ms=6, label='M-FULL P(high-risk)')
    ax_a.axhline(0.50, color=PALETTE['red'], lw=1.5, ls='--',
                 label='Operational threshold (P = 0.50)')

    # Annotate PHEIC and pandemic declarations
    pheic_x = covid_df[covid_df['date'] == '2020-01-30'].index
    if len(pheic_x):
        ax_a.axvline(list(x)[list(covid_df['date']).index('2020-01-30')],
                     color='orange', lw=1.2, ls=':', alpha=0.8)
        ax_a.text(list(x)[list(covid_df['date']).index('2020-01-30')] + 0.1,
                  0.55, 'PHEIC\ndeclared', fontsize=7.5, color='darkorange')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(date_labels, rotation=40, ha='right', fontsize=8)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel('M-FULL P(high-consequence)')
    ax_a.set_title('(a) M-FULL Detection Probability\nSARS-CoV-2 (Dec 2019 – Mar 2020)',
                   fontsize=9)
    ax_a.legend(fontsize=7.5, loc='upper left')

    # ── Panel (b): GCI and SMA ──
    ax_b = fig.add_subplot(gs[1])
    ax_b.plot(x, covid_df['GCI'], 's-', color=PALETTE['amber'],
              lw=2, ms=6, label='GCI (spatial concentration)')
    ax_b.plot(x, covid_df['SMA'], '^-', color=PALETTE['green'],
              lw=2, ms=6, label='SMA (seasonal anomaly)')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(date_labels, rotation=40, ha='right', fontsize=8)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_ylabel('Feature value (normalised [0,1])')
    ax_b.set_title('(b) GCI and SMA Signal Components\n(driving M-FULL detection)',
                   fontsize=9)
    ax_b.legend(fontsize=7.5, loc='center right')

    # Annotate GCI interpretation
    ax_b.annotate('GCI = 1.0\n(single country)\ndeclines as\nspread grows',
                  xy=(0, 1.0), xytext=(2.5, 1.08),
                  arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                  fontsize=7, ha='left')

    ax_a.text(-0.12, 1.04, '(a)', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')
    ax_b.text(-0.12, 1.04, '(b)', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    fig.suptitle('COVID-19 Early Detection Case Study — WHO DON Trajectory',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'covid19_probability_trace.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ covid19_probability_trace.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Supplementary validation experiments for the M-FULL paper.'
    )
    parser.add_argument('--data',   default='data/who_don_events.csv')
    parser.add_argument('--output', default='outputs/prospective/')
    parser.add_argument('--skip-covid', action='store_true',
                        help='Skip COVID-19 case study (faster run)')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Supplementary Validation | M-FULL Anti-Pandemic Pipeline")
    print("  The Lancet Digital Health | Ku Kang, 2026")
    print("=" * 65)

    # ── Load and prepare data ───────────────────────────────────────
    print("\n[1/4] Loading data and computing novel features…")
    df_raw  = load_and_preprocess(args.data)
    df_feat = add_novel_features(df_raw)
    df = df_feat.dropna(subset=FEAT_FULL + ['label']).copy()
    print(f"  Analytic cohort: n={len(df):,}  |  "
          f"High-consequence: n={df['label'].sum():,} ({df['label'].mean():.1%})")

    summary = {}

    # ── Experiment 1: Temporal holdout ────────────────────────────
    print("\n[2/4] Exp 1 — Pseudo-prospective temporal holdout…")
    holdout_results, roc_data = run_temporal_holdout(df)
    plot_temporal_roc(roc_data, out_dir)
    summary['temporal_holdout'] = holdout_results

    # ── Experiment 2: Feature integrity ablation ─────────────────
    print("\n[3/4] Exp 2 — Feature integrity ablation (5-fold CV × 5 seeds)…")
    integrity_results = run_feature_integrity(df)
    plot_feature_integrity(integrity_results, out_dir)
    summary['feature_integrity'] = integrity_results

    # ── Experiment 3: COVID-19 case study ────────────────────────
    if not args.skip_covid:
        print("\n[4/4] Exp 3 — COVID-19 early detection case study…")
        covid_results, covid_df = run_covid19_case_study(df, out_dir)
        plot_covid_trace(covid_df, out_dir)
        summary['covid19_case_study'] = covid_results
    else:
        print("\n[4/4] Skipping COVID-19 case study (--skip-covid)")

    # ── Save summary ──────────────────────────────────────────────
    with open(out_dir / 'prospective_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 65}")
    print(f"  All outputs saved to: {out_dir.resolve()}")
    print(f"  Files: temporal_holdout_roc.png | feature_integrity_bar.png")
    if not args.skip_covid:
        print(f"         covid19_probability_trace.png")
    print(f"         prospective_summary.json")
    print("=" * 65)


if __name__ == '__main__':
    main()
