-- Schema only. Applied once by the Supabase CLI.
-- Raw nhanes tables are created by the ingest pipeline from the XPT files.
-- ml tables are rebuilt from sql/derived on every load.

CREATE SCHEMA IF NOT EXISTS nhanes;
CREATE SCHEMA IF NOT EXISTS ml;
