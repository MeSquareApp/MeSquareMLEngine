# MeSquareMLEngine

Quantile biomarker prediction service for MeSquare. Trains LightGBM quantile models on NHANES-derived features and serves p10/p50/p90 predictions for `non_hdl`, `hdl`, `hba1c`, `egfr`, and `sbp`. The digital twin personalizes those predictions with lab results and wearable data, then scores 10-year cardiovascular risk with AHA PREVENT.

## Setup

```bash
uv sync
```

Copy `.env` with at least `DATABASE_URL` (and `NHANES_STORAGE_BASE_URL` for ingest). Prediction serving only needs trained artifacts under `artifacts/quantile/`.

## Run

```bash
uv run fastapi dev
```

Once the app runs, go to http://localhost:8000/

For documentation, go to http://localhost:8000/docs

## Test

```bash
uv run pytest
```

## Import NHANES data

Uses the same `DATABASE_URL` as MeSquareBE. This downloads the public NHANES `.xpt` files, reads them with pyreadstat, and loads one table per file into the `nhanes` schema. App tables stay in `public`.

After the raw load, the pipeline verifies the codebook row counts, builds temporary patient-feature tables, and stores `ml.cardiovascular_training`. Those temporary tables are dropped at the end of that load. Training code should read `ml.cardiovascular_training` and never the `.xpt` files. Schema creation lives in `supabase/migrations` and is applied once by the Supabase CLI. Raw table columns come from the XPT files, so the pipeline creates those tables itself.

```bash
uv run python upload_data.py
```

## Train models

Requires `DATABASE_URL` and a populated `ml.cardiovascular_training` table. Writes a versioned bundle under `artifacts/quantile/<version>/` (`model.joblib` + `metadata.json`).

```bash
uv run python train_model.py
```

## Predict

`POST /v1/predict` accepts a batch of patients and returns p10/p50/p90 for each target biomarker. The API loads the latest artifact on startup; `/health` reports `model_version`.

```bash
curl -s http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "patients": [{
      "sex": "Male",
      "age": 52,
      "race_eth": "Non-Hispanic White",
      "weight_kg": 88,
      "waist_cm": 101,
      "smoking": "former",
      "told_high_bp": 1,
      "bp_meds": 1,
      "told_high_chol": 0,
      "chol_meds": 0,
      "diabetes_dx": "no",
      "prediabetes_dx": 0,
      "insulin": 0,
      "diabetes_pills": 0
    }]
  }'
```

`sex` and `race_eth` accept NHANES labels (`Male` / `Female`, and the RIDRETH3 ethnicity strings). The API maps them to model codes via `src/constants`.

## Digital twin

`POST /v1/twin/predict`. The backend sends the user's profile (same fields as `/v1/predict`, plus `height_cm`), every lab result, and the end-of-day wearable aggregates from Open Wearables. The engine is stateless and recomputes the twin from that history on each call:

1. **Prior.** The latest quantile bundle gives population p10/p50/p90 for the profile.
2. **Labs.** A 1-D Kalman update per biomarker. Measurement noise is analytical plus within-person CV.
3. **Wearables.** The first 30 days set the user's own baseline. After that, the change in the 30-day trailing mean moves each biomarker through literature effect sizes and a first-order lag (`src/components/personalization/biomarker_config.py`).
4. **PREVENT.** Monte-Carlo samples from the biomarker posteriors go through the AHA PREVENT base equations, ported from the official package v1.1.0. The response has `population_prior`, `current`, and `forecast` 10-year risks.

Every trajectory point carries a `basis`:
- `prior`: no data yet
- `estimated`: wearables only
- `measured` / `anchored`: on or after a lab

It also carries a `confidence` label. `notebooks/personalization_twin.ipynb` prototypes and validates the full flow on simulated users.

`tests/test_prevent.py` checks the PREVENT coefficients against the official `prevent_beta10_2024.dta` when an AHA PREVENT checkout is found. Set `PREVENT_REPO_DIR`, or keep it as a sibling `PREVENT/` folder.

## Layout

```
config/                  environment settings
notebooks/               research prototypes (quantile prior, digital twin)
artifacts/quantile/      versioned model bundles (gitignored)
src/components/          load, transform, model train/eval/artifact, PREVENT
src/components/personalization/  Kalman filter, wearable features, biomarker settings
src/pipeline/            ingest, training, prediction, and twin workflows
src/repository/          Postgres connection and table loads
src/schemas/             API request/response models
src/utils/               logging, errors, and shared helpers
sql/derived/             patient features and training set, re-run on every ingest
supabase/migrations/     schema creation, applied once by the Supabase CLI
app.py                   launches the API
train_model.py           trains and saves quantile models
upload_data.py           pushes NHANES files into the database
tests/                   unit and API tests
```
