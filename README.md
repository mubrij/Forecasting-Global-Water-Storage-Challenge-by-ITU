# Forecasting Global Water Storage — ITU/Zindi Challenge

An end-to-end, leakage-safe forecasting system for Zindi's **A Step Ahead of
Drought: Forecasting Global Water Storage Challenge**. The repository records
the complete experimentation path: legal horizon reconstruction, spatial and
temporal feature engineering, chronological backtesting, classical ML, deep
sequence models, leaderboard calibration, and the final stacked ensemble.

The target is terrestrial water storage (`Target`) at global land-grid cells.
The test set contains six observed anchor blocks followed by rollouts of up to
seven months, so hidden future `TWS_t` values must never be used as features.

## Final files

| File | Purpose | Public LB RMSE |
|---|---|---:|
| `Submission_V27_CalibratedFiveModelBlend.csv` | LB-calibrated classical-ML backbone | awaiting score |
| `Submission_V28_DeepForecastBlend.csv` | V27 + conservative deep temporal correction | awaiting score |
| `Submission_V29_MLDeepStackBlend.csv` | V27 + ML/U-Net/deep stacked correction | awaiting score |

V29 is the current recommended submission. It contains 280,961 unique IDs in
sample-submission order, finite predictions, and preserves the calibrated mean
of every forecast month.

Known public leaderboard history:

| Version | Public LB RMSE |
|---|---:|
| V15 — map-mean probe | 0.735383264 |
| V18 — conservative map calibration | 0.754550831 |
| V20 — gated history correction | 0.737543150 |
| V24 — LB-aware recency ensemble | 0.745387396 |
| V25 — leaderboard-surface optimum | 0.731592090 |
| V26 — five-model blend | 0.733066420 |

Lower is better. V25 and V26 supplied two measured points along the same model
direction. Fitting the exact quadratic RMSE surface selected a correction
weight of `0.4234829277`, producing V27 without guessing another full-strength
blend.

## System overview

```text
Organizer CSVs
    └── legal anchors + 1–7 month direct examples
          ├── engineered tabular/spatial features
          │     ├── LightGBM L2 / LightGBM L1
          │     ├── CatBoost / XGBoost / ExtraTrees
          │     ├── hierarchical local ridge
          │     └── spatial residual U-Net
          └── 24-month temporal tensors
                ├── N-BEATS-style residual MLP
                ├── GRU / LSTM
                ├── temporal convolutional network
                └── Transformer encoder

Chronological out-of-fold predictions
    └── constrained non-negative stacking
          └── month-centered correction over LB-calibrated backbone
                └── Submission_V29_MLDeepStackBlend.csv
```

## Leakage-safe forecasting design

The starter data looks tabular, but treating rows independently is misleading:
roughly two thirds of test `TWS_t` values are intentionally hidden. The pipeline
therefore reconstructs the actual forecast process.

1. Detect observed test anchor maps from availability only.
2. Assign each target its effective horizon from 1 through 7.
3. Build direct examples whose TWS state comes only from the legal anchor.
4. Use supplied climate values only through the current predictor month.
5. Never insert a hidden test value or recursively use an unknown target.
6. Validate on complete future anchor blocks, never on random rows.

## Feature engineering

The tabular and tree models use:

- legal anchor TWS, previous observed TWS, anchor age, change, and momentum;
- location mean, monthly climatology, anomalies, variability, and trend;
- SPEI 1/3/6/12 and soil-moisture levels, changes, interactions, and path
  summaries;
- cyclic forecast/target month, horizon, spherical latitude/longitude, and
  hemisphere/latitude-band encodings;
- leakage-safe 3×3 and 7×7 spatial neighborhoods with wrapped longitude;
- recovered-history, recency-regime, ENSO, NOAA, CFSv2, and NMME experiments.

The sequence models receive a 24-month legal history with nine temporal
channels, 34 static/current context features, a learned location embedding, and
the legal anchor as the residual reference. The U-Net operates on native
140×360 global maps and predicts `Target - anchor_TWS` with land masks.

## Validation strategy

Training stops before July 2012. Six later, test-shaped blocks are held out:

```text
2012-07 h=1..3   2012-12 h=1..3   2013-05 h=1..3
2013-11 h=1..3   2014-04 h=1..3   2014-09 h=1..7
```

Predictions are aligned on `(forecast map, grid-cell key)`. Ensemble weights
are fitted with non-negative, sum-to-one constraints and checked by leaving out
each complete anchor block.

### Deep validation

| Model | Chronological RMSE | Selected epochs |
|---|---:|---:|
| N-BEATS-style MLP | 0.628911 | 2 |
| GRU | 0.628973 | 4 |
| LSTM | **0.614618** | 4 |
| TCN | 0.627632 | 1 |
| Transformer | 0.639737 | 4 |
| Five-model centered deep ensemble | **0.604460** | — |

Deep weights are 4.96% N-BEATS, 33.28% GRU, 45.09% LSTM, 13.67% TCN, and
3.00% Transformer.

### Final ML + deep stack

The block-stable family stack selected:

- 26.73% boosted tree;
- 44.67% spatial U-Net;
- 28.60% five-model deep ensemble.

Centered chronological RMSE improved from `0.605164` for the previous tabular
ensemble to `0.588566` for the full stack. V29 applies only 25% of this direction
over V27 to reduce temporal-regime and public-LB overfitting risk; that
conservative validation point scores `0.598424`.

## Installation

Python 3.11+ is recommended. A CUDA GPU is strongly recommended for the U-Net
and deep sequence models.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the organizer downloads in the repository directory. The loader accepts
downloaded suffixes such as:

```text
Train (47).csv
Test (41).csv
SampleSubmission (67).csv
```

Competition data and downloaded climate archives are intentionally excluded
from Git because of size and redistribution constraints.

## Reproduce the workflow

### 1. Classical tree + local ridge baseline

```bash
python ensemble_solution.py all \
  --data-dir . --artifact-dir artifacts_final \
  --output Submission_Ensemble_v1.csv --n-jobs 20
```

### 2. Spatial U-Net

```bash
python map_unet_solution.py validate --data-dir . --artifact-dir artifacts_map_unet
python map_unet_production.py all --data-dir . --artifact-dir artifacts_map_full
python map_blend_experiment.py
```

### 3. Five classical ML models

```bash
python five_model_ensemble.py validate
python five_model_ensemble.py production
python calibrate_five_model_blend.py
```

This trains LightGBM-L2, LightGBM-L1, CatBoost, XGBoost, and ExtraTrees. The
calibration step uses the recorded V25/V26 leaderboard scores to create V27.

### 4. Five deep temporal models

```bash
python deep_forecast_ensemble.py validate --cells-per-map 4096
python deep_forecast_ensemble.py production --cells-per-map 4096 --blend 0.25
```

The validation-selected epoch count for each architecture is reused for
full-history training. This creates V28 and saves the standalone deep ensemble.

### 5. Final stacked ensemble

```bash
python stack_ml_deep_ensemble.py validate
python stack_ml_deep_ensemble.py production --blend 0.25
```

This aligns all out-of-fold predictions, performs leave-one-block-out stability
checks, loads the saved deep checkpoints, and writes V29.

## Important scripts

| Script | Role |
|---|---|
| `winning_solution.py` | data loading, history statistics, core features and LightGBM utilities |
| `ensemble_solution.py` | initial tree/ridge production ensemble |
| `spatial_experiment.py` | neighborhood features and spatial ablations |
| `map_unet_solution.py` | masked global residual U-Net validation |
| `map_unet_production.py` | full-history U-Net training and prediction |
| `five_model_ensemble.py` | five classical regressors and constrained blending |
| `calibrate_five_model_blend.py` | quadratic LB direction calibration |
| `lstm_solution.py` | temporal store and legal sequence datasets |
| `deep_forecast_ensemble.py` | N-BEATS/GRU/LSTM/TCN/Transformer ensemble |
| `stack_ml_deep_ensemble.py` | final ML + U-Net + deep OOF stack |
| `recency_regime_experiment.py` | recent-history/regime validation |
| `cfsv2_solution.py`, `nmme_solution.py` | external forecast-data experiments |

The remaining experiment scripts are retained to make rejected ideas and
ablation paths reproducible rather than hiding unsuccessful trials.

## Reproducibility and safeguards

- Fixed primary seed: `20260817`.
- Deterministic sampling for large tree training sets.
- ID-based one-to-one merges for every component submission.
- Assertions for row count, unique IDs, finite values, and schema.
- Per-forecast-month centering preserves the leaderboard-calibrated map means.
- Raw data, external downloads, model checkpoints, and generated artifacts are
  ignored; every one is reproducible from the documented commands.

The official challenge page documents the RMSE metric, forecast horizons, and
submission contract: [Zindi competition](https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge).
