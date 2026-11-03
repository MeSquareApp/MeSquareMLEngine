# MeSquareMLEngine

Blank FastAPI template.

## Setup

```bash
uv sync
```

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

After the raw load, the pipeline verifies the codebook row counts, builds temporary patient-feature tables, and stores `ml.cardiovascular_training`. Those temporary tables are dropped at the end of that load. Training code should read `ml.cardiovascular_training` (via `load_cardiovascular_training`) and never the `.xpt` files. Schema creation lives in `supabase/migrations` and is applied once by the Supabase CLI. Raw table columns come from the XPT files, so the pipeline creates those tables itself.

```bash
uv run python upload_data.py
```

## Train the population prior

Trains the LightGBM quantile models (non-HDL, HDL, HbA1c, eGFR, SBP) from `ml.cardiovascular_training` and saves them to `artifacts/prior/` (gitignored). The API loads them at startup.

```bash
uv run python train_prior.py
```

## Digital twin API

`POST /twin/predict`. The backend sends the user's profile, every lab result, and the end-of-day wearable aggregates from Open Wearables. The engine is stateless and recomputes the twin from that history on each call:

1. **Prior.** Population quantiles for the profile.
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
notebooks/               quantile prior exploration, digital twin prototype
src/components/          load, transform, prior model, PREVENT
src/components/personalization/  Kalman filter, wearable features, biomarker settings
src/pipeline/            ingest, prior training, and twin workflows
src/schemas/             API request and response models
src/repository/          Postgres connection and table loads
src/utils/               logging, errors, and shared helpers
sql/derived/             patient features and training set, re-run on every ingest
supabase/migrations/     schema creation, applied once by the Supabase CLI
app.py                   launches the API
upload_data.py           pushes NHANES files into the database
train_prior.py           trains and saves the population prior
tests/                   unit and API tests
```
