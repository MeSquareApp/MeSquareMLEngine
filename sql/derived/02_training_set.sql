-- ml.cardiovascular_training. The pipeline re-runs this after 01_patient_features.sql.
-- The table is dropped, created, then filled. A second run replaces the rows.

DROP TABLE IF EXISTS ml.cardiovascular_training;

CREATE TABLE ml.cardiovascular_training (
    seqn double precision PRIMARY KEY,
    sex double precision,
    age double precision,
    race_eth double precision,
    sdmvstra double precision,
    sdmvpsu double precision,
    height_cm double precision,
    weight_kg double precision,
    bmi double precision,
    waist_cm double precision,
    sbp double precision,
    dbp double precision,
    pulse double precision,
    smoking text,
    told_high_bp integer,
    bp_meds integer,
    told_high_chol integer,
    chol_meds integer,
    diabetes_dx text,
    prediabetes_dx integer,
    insulin integer,
    diabetes_pills integer,
    non_hdl double precision,
    hdl double precision,
    hba1c double precision,
    scr double precision,
    wt_lab double precision
);

INSERT INTO ml.cardiovascular_training (
    seqn,
    sex,
    age,
    race_eth,
    sdmvstra,
    sdmvpsu,
    height_cm,
    weight_kg,
    bmi,
    waist_cm,
    sbp,
    dbp,
    pulse,
    smoking,
    told_high_bp,
    bp_meds,
    told_high_chol,
    chol_meds,
    diabetes_dx,
    prediabetes_dx,
    insulin,
    diabetes_pills,
    non_hdl,
    hdl,
    hba1c,
    scr,
    wt_lab
)
WITH bp AS (
    SELECT
        seqn,
        (
            SELECT round(avg(v)::numeric, 2)
            FROM unnest(ARRAY[bpxosy1, bpxosy2, bpxosy3]) AS v
        ) AS sbp,
        (
            SELECT round(avg(v)::numeric, 2)
            FROM unnest(ARRAY[bpxodi1, bpxodi2, bpxodi3]) AS v
        ) AS dbp,
        (
            SELECT round(avg(v)::numeric, 2)
            FROM unnest(ARRAY[bpxopls1, bpxopls2, bpxopls3]) AS v
        ) AS pulse
    FROM bpxo
)
SELECT
    d.seqn,
    d.sex,
    d.age,
    d.race_eth,
    d.sdmvstra,
    d.sdmvpsu,
    b.bmxht,
    b.bmxwt,
    b.bmxbmi,
    b.bmxwaist,
    bp.sbp,
    bp.dbp,
    bp.pulse,
    CASE
        WHEN s.smq020 = 2 THEN 'never'
        WHEN s.smq020 = 1 AND s.smq040 IN (1, 2) THEN 'current'
        WHEN s.smq020 = 1 AND s.smq040 = 3 THEN 'former'
    END,
    CASE q.bpq020 WHEN 1 THEN 1 WHEN 2 THEN 0 END,
    CASE
        WHEN q.bp_meds_q = 1 THEN 1
        WHEN q.bp_meds_q = 2 OR q.bpq020 = 2 THEN 0
    END,
    CASE q.bpq080 WHEN 1 THEN 1 WHEN 2 THEN 0 END,
    CASE q.chol_meds_q WHEN 1 THEN 1 WHEN 2 THEN 0 END,
    CASE i.diq010
        WHEN 1 THEN 'yes'
        WHEN 3 THEN 'borderline'
        WHEN 2 THEN 'no'
    END,
    CASE i.diq160 WHEN 1 THEN 1 WHEN 2 THEN 0 END,
    CASE
        WHEN i.diq050 = 1 THEN 1
        WHEN i.diq050 = 2 OR i.diq010 IN (2, 3) THEN 0
    END,
    CASE
        WHEN i.diq070 = 1 THEN 1
        WHEN i.diq070 = 2 THEN 0
        WHEN i.diq010 = 2 AND coalesce(i.diq160, 2) <> 1 THEN 0
    END,
    CASE
        WHEN l.lbxtc IS NOT NULL AND l.lbdhdd IS NOT NULL THEN l.lbxtc - l.lbdhdd
    END,
    l.lbdhdd,
    l.lbxgh,
    l.lbxscr,
    l.wt_lab
FROM demographics d
LEFT JOIN bmx b USING (seqn)
LEFT JOIN bp USING (seqn)
LEFT JOIN smq s USING (seqn)
LEFT JOIN bpq q USING (seqn)
LEFT JOIN diq i USING (seqn)
LEFT JOIN labs l USING (seqn)
WHERE d.ridstatr = 2
  AND d.age BETWEEN 20 AND 79
  AND coalesce(d.ridexprg, 2) <> 1;
