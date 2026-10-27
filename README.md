# MeSquareMLEngine

Quantile biomarker prediction service for MeSquare. Trains LightGBM quantile models on NHANES-derived features and serves p10/p50/p90 predictions for `non_hdl`, `hdl`, `hba1c`, and `egfr`.

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

## Layout

```
config/                  environment settings
notebooks/               research prototypes
artifacts/quantile/      versioned model bundles (gitignored)
src/components/          load, transform, model train/eval/artifact
src/pipeline/            ingest, training, and prediction workflows
src/repository/          Postgres connection and table loads
src/schemas/             API request/response models
src/utils/               logging, errors, and shared helpers
sql/derived/             patient features and training set, re-run on every ingest
supabase/migrations/     schema creation, applied once by the Supabase CLI
app.py                   launches the API
train_model.py           trains and saves quantile models
upload_data.py           pushes NHANES files into the database
```
