-- Wearable-adjusted predictions computed in shadow mode. They are never served or
-- fed to PREVENT until the analyte's wearable stage is activated; this table is the
-- evidence for that decision. One row per user, analyte, mode, day and input hash.

CREATE TABLE IF NOT EXISTS ml.shadow_predictions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id text NOT NULL,
    analyte text NOT NULL,
    mode text NOT NULL CHECK (mode IN ('level', 'change')),
    as_of date NOT NULL,
    -- (p10, p50, p90) in the analyte's canonical unit.
    served_band double precision[] NOT NULL CHECK (cardinality(served_band) = 3),
    shadow_band double precision[] NOT NULL CHECK (cardinality(shadow_band) = 3),
    residual_model_version text NOT NULL,
    prior_version text NOT NULL,
    input_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, analyte, mode, as_of, input_hash)
);

CREATE INDEX IF NOT EXISTS shadow_predictions_user_as_of
    ON ml.shadow_predictions (user_id, as_of);
