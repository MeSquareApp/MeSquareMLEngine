-- Temporary patient-feature tables. They exist only while ml.cardiovascular_training is built.
-- ON COMMIT DROP removes them when that transaction commits.

DROP TABLE IF EXISTS ml.demo;
DROP TABLE IF EXISTS ml.demographics;
DROP TABLE IF EXISTS demographics;

CREATE TEMP TABLE demographics (
    seqn double precision PRIMARY KEY,
    sex double precision,
    age double precision,
    race_eth double precision,
    ridstatr double precision,
    ridexprg double precision,
    wt_mec double precision,
    sdmvstra double precision,
    sdmvpsu double precision
) ON COMMIT DROP;

INSERT INTO demographics (
    seqn,
    sex,
    age,
    race_eth,
    ridstatr,
    ridexprg,
    wt_mec,
    sdmvstra,
    sdmvpsu
)
SELECT
    seqn,
    riagendr,
    ridageyr,
    ridreth3,
    ridstatr,
    ridexprg,
    wtmec2yr,
    sdmvstra,
    sdmvpsu
FROM nhanes.demographics;

DROP TABLE IF EXISTS ml.bpq;
DROP TABLE IF EXISTS bpq;

CREATE TEMP TABLE bpq (
    seqn double precision PRIMARY KEY,
    bpq020 double precision,
    bp_meds_q double precision,
    bpq080 double precision,
    chol_meds_q double precision
) ON COMMIT DROP;

INSERT INTO bpq (
    seqn,
    bpq020,
    bp_meds_q,
    bpq080,
    chol_meds_q
)
SELECT
    seqn,
    bpq020,
    bpq150,
    bpq080,
    bpq101d
FROM nhanes.blood_pressure_cholesterol_questionnaires;

DROP TABLE IF EXISTS ml.bmx;
DROP TABLE IF EXISTS bmx;

CREATE TEMP TABLE bmx (
    seqn double precision PRIMARY KEY,
    bmxht double precision,
    bmxwt double precision,
    bmxbmi double precision,
    bmxwaist double precision
) ON COMMIT DROP;

INSERT INTO bmx (
    seqn,
    bmxht,
    bmxwt,
    bmxbmi,
    bmxwaist
)
SELECT
    seqn,
    bmxht,
    bmxwt,
    bmxbmi,
    bmxwaist
FROM nhanes.body_measures;

DROP TABLE IF EXISTS ml.bpxo;
DROP TABLE IF EXISTS bpxo;

CREATE TEMP TABLE bpxo (
    seqn double precision PRIMARY KEY,
    bpxosy1 double precision,
    bpxosy2 double precision,
    bpxosy3 double precision,
    bpxodi1 double precision,
    bpxodi2 double precision,
    bpxodi3 double precision,
    bpxopls1 double precision,
    bpxopls2 double precision,
    bpxopls3 double precision
) ON COMMIT DROP;

INSERT INTO bpxo (
    seqn,
    bpxosy1,
    bpxosy2,
    bpxosy3,
    bpxodi1,
    bpxodi2,
    bpxodi3,
    bpxopls1,
    bpxopls2,
    bpxopls3
)
SELECT
    seqn,
    bpxosy1,
    bpxosy2,
    bpxosy3,
    bpxodi1,
    bpxodi2,
    bpxodi3,
    bpxopls1,
    bpxopls2,
    bpxopls3
FROM nhanes.blood_pressure_readings;

DROP TABLE IF EXISTS ml.smq;
DROP TABLE IF EXISTS smq;

CREATE TEMP TABLE smq (
    seqn double precision PRIMARY KEY,
    smq020 double precision,
    smq040 double precision
) ON COMMIT DROP;

INSERT INTO smq (
    seqn,
    smq020,
    smq040
)
SELECT
    seqn,
    smq020,
    smq040
FROM nhanes.smoking_questionnaires;

DROP TABLE IF EXISTS ml.diq;
DROP TABLE IF EXISTS diq;

CREATE TEMP TABLE diq (
    seqn double precision PRIMARY KEY,
    diq010 double precision,
    diq160 double precision,
    diq050 double precision,
    diq070 double precision
) ON COMMIT DROP;

INSERT INTO diq (
    seqn,
    diq010,
    diq160,
    diq050,
    diq070
)
SELECT
    seqn,
    diq010,
    diq160,
    diq050,
    diq070
FROM nhanes.diabetes_questionnaires;

DROP TABLE IF EXISTS ml.labs;
DROP TABLE IF EXISTS labs;

CREATE TEMP TABLE labs (
    seqn double precision PRIMARY KEY,
    lbxtc double precision,
    lbdhdd double precision,
    lbxgh double precision,
    lbxscr double precision,
    wt_lab double precision
) ON COMMIT DROP;

INSERT INTO labs (
    seqn,
    lbxtc,
    lbdhdd,
    lbxgh,
    lbxscr,
    wt_lab
)
WITH persons AS (
    SELECT seqn FROM nhanes.hdl_cholesterol_results
    UNION
    SELECT seqn FROM nhanes.total_cholesterol_results
    UNION
    SELECT seqn FROM nhanes.glycohemoglobin_results
    UNION
    SELECT seqn FROM nhanes.biochemistry_profiles
)
SELECT
    p.seqn,
    t.lbxtc,
    h.lbdhdd,
    NULLIF(g.lbxgh, 0),
    b.lbxscr,
    COALESCE(h.wtph2yr, t.wtph2yr, g.wtph2yr, b.wtph2yr)
FROM persons p
LEFT JOIN nhanes.hdl_cholesterol_results h USING (seqn)
LEFT JOIN nhanes.total_cholesterol_results t USING (seqn)
LEFT JOIN nhanes.glycohemoglobin_results g USING (seqn)
LEFT JOIN nhanes.biochemistry_profiles b USING (seqn);
