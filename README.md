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

```bash
uv run python upload_data.py
```

## Layout

```
config/                  environment settings
notebooks/
src/components/          load and transform source data
src/pipeline/            orchestrates the ingest workflow
src/repository/          Postgres connection and table loads
src/utils/               logging, errors, and shared helpers
app.py                   launches the API
upload_data.py           pushes NHANES files into the database
```
