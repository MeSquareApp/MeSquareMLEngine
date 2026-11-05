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

### Run with Docker

The image installs R 4.6+ from CRAN (`bookworm-cran46`), `rpy2`, and `AHAprevent` from `PREVENT/AHAprevent_1.1.0.tar.gz`, then serves the API. Mount trained model artifacts from the host; set `DATABASE_URL` in `.env` only if you need shadow prediction writes.

```bash
docker build -t mesquare-mlengine .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/artifacts:/app/artifacts" \
  mesquare-mlengine
```

Open http://localhost:8000/docs. `GET /health` should report a PREVENT version such as `AHAprevent 1.1.0`, which confirms R and rpy2 loaded the package.

## Test

```bash
uv run pytest
```

Unit tests mock external dependencies such as R, the model files and the database. `tests/test_prevent_official.py` runs against the installed AHAprevent R package and skips when it is missing.

Lint, format and type-check before committing (see `Claude.md`):

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src app.py
```

Notebooks need a Jupyter kernel in `.venv`. It isn't a project requirement:

```bash
uv pip install ipykernel ipywidgets
```

## Import NHANES data

Uses the same `DATABASE_URL` as MeSquareBE. This downloads the public NHANES `.xpt` files, reads them with pyreadstat, and loads one table per file into the `nhanes` schema. App tables stay in `public`.

After the raw load, the pipeline verifies the codebook row counts, builds temporary patient-feature tables, and stores `ml.cardiovascular_training`. Those temporary tables are dropped at the end of that load. Training code should read `ml.cardiovascular_training` and never the `.xpt` files. Schema creation lives in `supabase/migrations` and is applied once by the Supabase CLI. Raw table columns come from the XPT files, so the pipeline creates those tables itself.

```bash
uv run python upload_data.py
```

## Train models

Requires `DATABASE_URL` and a populated `ml.cardiovascular_training` table. Writes a versioned bundle under `artifacts/quantile/<version>/` (`model.joblib` + `metadata.json`).

How training works:
- **Survey weights.** `wt_lab` for the blood biomarkers and `wt_mec` (joined from `nhanes.demographics`) for SBP, in fitting, early stopping, CV and every reported metric. Quantiles and coverage describe US adults, not the oversampled NHANES sample.
- **Grouped splits.** The holdout and the CV folds are split by NHANES design cluster (stratum x PSU), so people from one sampled area are never on both sides.
- **Outlier rule.** Target values outside the plausible ranges in `src/components/biomarkers.py` are dropped. These are the same limits lab ingestion uses.
- **eGFR.** Race-free CKD-EPI 2021, the equation AHA PREVENT was derived with.
- **Calibration.** The p10-p90 band is calibrated per sex x age band (Mondrian conformal) on out-of-fold predictions. The widening is stored in the bundle and applied by both endpoints. Coverage per subgroup is reported cross-conformally on every training row and on the held-out clusters.

Then judge the lab stage by simulation and write the activation registry (`config/activation.json`, committed):

```bash
uv run python train_model.py
uv run python evaluate_lab_stage.py
```

## API

Both endpoints return biomarker bands and a cardiovascular risk score in one response. Every response carries `model_versions` and an `input_hash` (SHA-256 of the canonical inputs and model versions; record order does not matter). `/health` reports the prior and PREVENT versions.

### `POST /v1/predict`: population prior only

A batch of profiles (`sex` / `race_eth` take NHANES labels, such as `Male` or `Non-Hispanic White`). Each patient gets:
- p10/p50/p90 per biomarker, all with `basis: "population"`;
- PREVENT run on the predicted p50 values, with `risk_basis: "estimated"`.

PREVENT needs `height_cm` and `weight_kg`. Without them, `risk.status` is `not_generated` and `missing_inputs` lists them.

### `POST /v1/twin/state`: personalized state

The backend sends `profile`, every `records` entry (labs, home BP, wearable daily summaries) and optionally `as_of` and `user_id`. The engine is stateless. Records use one shape:

```json
{"id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "biomarker_abbr": "HDL", "value": 52,
 "unit": "mg/dL", "measurement_method": "blood_test_lab_report_upload",
 "measured_at": "2026-11-03T20:50:55.898Z"}
```

How records are handled:
- **Mapping.** Each record maps to a canonical biomarker through one alias table (`src/components/biomarkers.py`).
- **Units.** Values are converted to the canonical unit (mmol/L, mmol/mol, µmol/L and lb are accepted).
- **Checks.** Values must be inside plausibility limits. Lab values must come from a measured source (`blood_test_lab_report_upload`, `lab_integration`, `clinic_measurement`, `home_bp_protocol`). Wearable values must come from a `wearable_*` or `smart_scale*` source.
- **Rejections.** Anything else is listed in `rejected_records` with a reason, and logged.
- **Derived values.** The engine computes non-HDL (total cholesterol minus HDL, same draw) and eGFR (CKD-EPI 2021). A value reported directly that day wins over a derived one.

Per biomarker, the served band follows the stage the user has reached:

| Stage | User has | Band | `basis` |
|---|---|---|---|
| 0 | profile only | population quantile prior | `population` |
| 1 | a lab | centred on the latest lab; width = sqrt(2) x published within-person variation, widening with the lab's age | `lab` |
| 2 | 2+ labs | stage 1, plus `change`: latest vs previous against the 95% reference change value | `lab` |
| 3 | wearables (after activation) | stage 0 or 1 plus a residual adjustment | `population_wearable` / `lab_wearable` |

Each band carries `basis_label` for the app ("Typical for your profile", "Based on your Nov 1 lab"), plus `source_lab_id` when lab-based.

Rules:
- **Anchor reset.** Starting a statin (`statin_started_at`) returns the lipids to the population prior until a new lab arrives. Starting BP medication (`bp_meds_started_at`) does the same for SBP, and `pregnant` does it for every biomarker.
- **Never worse.** A stage is served only where `config/activation.json` says it beat the stage below on evaluation.
- **Risk.** PREVENT uses measured labs where present and the personalized p50 otherwise.
  - `inputs_used` gives each input's value and source (`measured`, `predicted`, `reported`).
  - `risk_basis` is `measured`, `mixed` or `estimated`.
  - Results come per outcome (total CVD, ASCVD, heart failure) and horizon: 10 years, plus 30 years for ages 30-59. Each has `risk_pct`, `risk_low`, `risk_high`, and `category` for 10-year total CVD.
  - The range is PREVENT at the p10/p90 ends of every predicted input.
  - Inputs outside AHA's validated ranges are clipped and listed in `out_of_range`.
  - Pregnancy, age outside 30-79, or a missing R package give `not_generated` with a `reason`.
- **Lab prompt.** `lab_prompt` says when to ask for the next lab: 90 days after the latest lab, or now.
- **Wearables.** Daily summaries of resting heart rate, HRV (RMSSD), VO2 max and scale weight become 30- and 90-day windows: mean, SD, Theil-Sen slope and coverage. A window needs at least 70% day coverage.
  - A residual model (`level` mode for wearables-only users, `change` mode 60+ days after a lab) runs in **shadow mode**. With a `user_id`, its bands are written to `ml.shadow_predictions` (`supabase/migrations/20261103180000_create_shadow_predictions.sql`) and never returned or fed to PREVENT.
  - It is served only after `gate_residual_model()` passes on real consented pairs: at least 200 users, at least 10% lower pinball loss than both last lab and population prior on held-out users, a bootstrap 95% CI excluding zero, and 75-85% coverage.
  - Activation order is HbA1c, then HDL and non-HDL. eGFR stays lab-only.

The app should label any risk whose `risk_basis` is not `measured` as an estimate, for example "Estimated — upload a lab for a measured score".

**When to call it.** The backend recomputes the state on each update trigger: a new lab, a profile change (weight, smoking, medication, diagnosis), a completed home BP protocol, or a new active model version. A changed `model_versions` means the last one. It also calls nightly after the Open Wearables sync, with `user_id`, so shadow predictions accumulate. An unchanged `input_hash` means nothing changed, so the response is identical.

### AHA PREVENT (R) setup

PREVENT results require R and the official package from the AHA-DS-Analytics/PREVENT checkout in this repo (`PREVENT/`). If either is missing, the API still answers, with `risk.status: "not_generated"` and a `reason`.

```bash
brew install r
R CMD INSTALL PREVENT/AHAprevent_1.1.0.tar.gz
```

On a Linux server, install R (`r-base`) and run the same `R CMD INSTALL`. The Docker image does this for you. `rpy2` finds R through `R_HOME` or the `R` on `PATH`.

The package is GPL-3. Have whoever handles licensing review how it ships with the service.

`tests/test_prevent.py` checks the installed package against a published reference case. It also checks it against the official `prevent_beta10_2024.dta` and `prevent_test.dta` when the PREVENT folder is present. Set `PREVENT_REPO_DIR` to override the default `PREVENT/` path.

## Goal status

Against the goals in "MeSquare Personalization Goals". Simulation numbers come from `config/activation.json` (published within-person variation, NHANES held-out people as true levels). Prior numbers come from the bundle metrics.

| # | Goal | Status now |
|---|---|---|
| 1 | Next-lab error at least 40% lower than population | Simulation: met for non-HDL (53%), HDL (51%), eGFR (48%); not for HbA1c (29%) or SBP (26%). Real check needs 2+ labs per user |
| 2 | 80% band covers 75-85% | Met. Prior on held-out clusters 0.79-0.82; lab band in simulation 0.80-0.82 |
| 3 | Lab band at most 1/3 of population width | Not reachable with published variation. Simulated ratios 0.45-0.83; even a same-day lab gives 0.37-0.75 |
| 4 | Coverage gap at most 5 points across sex and age bands | Met. Cross-conformal gaps 0.0-0.7 points (held-out clusters are noisier, a few hundred people per group) |
| 5 | False "real change" alerts at most 5% | Simulation 4.3-5.7% (5% expected by design; SBP's 5.7% is within sampling noise) |
| 6 | Wearables earn their place | Gate built and tested (planted signal passes, null fails). Awaits real consented pairs; all wearable stages off |
| 7 | Stable, replayable | Met. Deterministic output, order-independent `input_hash`, replay tests |
| 8 | Correct risk | Met. Official AHAprevent 1.1.0 through rpy2, checked against its beta table (1e-6) and a published case |
| 9 | Everyone gets a result, p95 under 300 ms | Met. Every response has bands and a risk block. Live p95 was 9 ms (`/v1/twin/state`) and 42 ms (`/v1/predict`) |
| 10 | Traceable and explainable | Met. `model_versions`, `input_hash`, `basis`, `basis_label`, `source_lab_id`, `inputs_used` |

**Quick win tested.** Adding clinic pulse to the prior changed holdout pinball loss by -0.3% to +0.5% per target, so it was not added.

### Open questions: decisions taken

- **Estimated risk.** Returned, labelled by `risk_basis` with a range. This differs from v2's measured-only design, so the team should still confirm it.
- **SBP without a cuff.** Predicted SBP is used and marked `predicted` in `inputs_used`. A `home_bp_protocol` reading makes it measured.
- **Range method.** The p10/p90 plug-in, as the API table describes: deterministic (goal 7) and fast (goal 9). Monte Carlo was replaced.
- **Abbreviations.** One alias table in `src/components/biomarkers.py`. Add the LlamaIndex extractor's exact names there once confirmed.
- **Simulation data.** Published within-person variation (`src/components/personalization/biomarker_config.py`), as `simulation.py` documents.

## Layout

```
config/                  environment settings
notebooks/               research prototypes (quantile prior, digital twin)
artifacts/quantile/      versioned model bundles (gitignored)
artifacts/residual/      wearable residual models, once trained on real pairs (gitignored)
config/activation.json   which personalization stages are served, with evidence
src/components/          load, transform, biomarkers, record ingestion, calibration,
                         model train/eval/artifact, PREVENT, risk
src/components/personalization/  lab anchor, wearable windows, residual model and gate,
                         activation registry, simulation
src/pipeline/            ingest, training, evaluation, prediction, and twin workflows
src/repository/          Postgres connection, table loads, shadow prediction store
src/schemas/             API request/response models
src/utils/               logging, errors, and shared helpers
sql/derived/             patient features and training set, re-run on every ingest
supabase/migrations/     schema creation, applied once by the Supabase CLI
PREVENT/                 official AHAprevent R package and reference tables
Dockerfile               API image with R, rpy2, and AHAprevent
app.py                   launches the API
train_model.py           trains and saves quantile models
evaluate_lab_stage.py    simulates repeat labs and writes the activation registry
upload_data.py           pushes NHANES files into the database
tests/                   unit and API tests
```
