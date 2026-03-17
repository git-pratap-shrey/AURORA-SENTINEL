-- Migration: Add Two-Tier Scoring Fields to Alerts Table
-- Date: 2026-03-03
-- Description: Adds ML/AI score fields for enhanced fight detection

-- Add new columns for two-tier scoring
ALTER TABLE alerts ADD COLUMN ml_score REAL;
ALTER TABLE alerts ADD COLUMN ai_score REAL;
ALTER TABLE alerts ADD COLUMN final_score REAL;
ALTER TABLE alerts ADD COLUMN detection_source TEXT;
ALTER TABLE alerts ADD COLUMN ai_explanation TEXT;
ALTER TABLE alerts ADD COLUMN ai_scene_type TEXT;
ALTER TABLE alerts ADD COLUMN ai_confidence REAL;

-- Update existing alerts to populate new fields from legacy risk_score
UPDATE alerts SET 
    ml_score = risk_score,
    ai_score = 0.0,
    final_score = risk_score,
    detection_source = 'ml',
    ai_scene_type = 'normal'
WHERE ml_score IS NULL;

-- Create index on final_score for faster queries
CREATE INDEX idx_alerts_final_score ON alerts(final_score);
CREATE INDEX idx_alerts_detection_source ON alerts(detection_source);
