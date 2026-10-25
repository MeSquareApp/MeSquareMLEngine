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

## Layout

```
config/                  environment settings
notebooks/
src/components/          load, transform
src/pipeline/            orchestrates the ingest workflow
src/repository/          Postgres connection and table loads
src/utils/               logging, errors, and shared helpers
sql/derived/             patient features and training set, re-run on every ingest
supabase/migrations/     schema creation, applied once by the Supabase CLI
app.py                   launches the API
upload_data.py           pushes NHANES files into the database
```
