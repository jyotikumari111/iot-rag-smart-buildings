-- Timescale hypertable for telemetry
CREATE TABLE IF NOT EXISTS telemetry (
  site_id   TEXT NOT NULL,
  equip_id  TEXT NOT NULL,
  sensor    TEXT NOT NULL,
  ts        TIMESTAMPTZ NOT NULL,
  value     DOUBLE PRECISION,
  unit      TEXT,
  status    TEXT DEFAULT 'OK'
);

-- Install Timescale extension if available
CREATE EXTENSION IF NOT EXISTS timescaledb;
SELECT create_hypertable('telemetry', by_range('ts'), if_not_exists => TRUE);

-- Helpful index
CREATE INDEX IF NOT EXISTS idx_telemetry_key ON telemetry (site_id, equip_id, sensor, ts DESC);
