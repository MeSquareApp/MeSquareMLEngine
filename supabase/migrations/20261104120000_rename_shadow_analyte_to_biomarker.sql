-- Align shadow_predictions with the engine contract: analyte -> biomarker.

ALTER TABLE ml.shadow_predictions
    RENAME COLUMN analyte TO biomarker;

ALTER TABLE ml.shadow_predictions
    DROP CONSTRAINT IF EXISTS shadow_predictions_user_id_analyte_mode_as_of_input_hash_key;

ALTER TABLE ml.shadow_predictions
    ADD CONSTRAINT shadow_predictions_user_id_biomarker_mode_as_of_input_hash_key
    UNIQUE (user_id, biomarker, mode, as_of, input_hash);
