-- Facial Image Warping - PostgreSQL Schema
-- Run once on the remote server: psql -U senguser -d sengproject -f migrate.sql

CREATE TABLE IF NOT EXISTS users (
  id         SERIAL PRIMARY KEY,
  email      TEXT        NOT NULL UNIQUE,
  username   TEXT        NOT NULL UNIQUE,
  password   TEXT        NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
  id         SERIAL PRIMARY KEY,
  user_id    INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  file_name  TEXT        NOT NULL,
  image_uri  TEXT        NOT NULL,
  width      INTEGER     NOT NULL,
  height     INTEGER     NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS preprocess_runs (
  id              SERIAL PRIMARY KEY,
  user_id         INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  upload_id       INTEGER     NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  face_detected   BOOLEAN     NOT NULL,
  cropped         BOOLEAN     NOT NULL,
  normalized      BOOLEAN     NOT NULL,
  grayscale_ready BOOLEAN     NOT NULL,
  width           INTEGER     NOT NULL,
  height          INTEGER     NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS landmark_sets (
  id             SERIAL PRIMARY KEY,
  user_id        INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  upload_id      INTEGER     NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  source         TEXT        NOT NULL,
  count          INTEGER     NOT NULL,
  landmarks_json TEXT        NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS warp_runs (
  id                   SERIAL PRIMARY KEY,
  user_id              INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  upload_id            INTEGER     NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  mode                 TEXT        NOT NULL,
  ai_model             TEXT        NOT NULL,
  ai_transfer_enabled  BOOLEAN     NOT NULL,
  expression_intensity INTEGER     NOT NULL,
  aging_level          INTEGER     NOT NULL,
  smoothing_level      INTEGER     NOT NULL,
  transformed_ready    BOOLEAN     NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS frequency_runs (
  id                       SERIAL PRIMARY KEY,
  user_id                  INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  upload_id                INTEGER     NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  high_frequency_energy    REAL        NOT NULL,
  low_frequency_energy     REAL        NOT NULL,
  fourier_ready            BOOLEAN     NOT NULL,
  magnitude_spectrum_ready BOOLEAN     NOT NULL,
  created_at               TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
  id         SERIAL PRIMARY KEY,
  user_id    INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  upload_id  INTEGER     NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  mse        REAL        NOT NULL,
  psnr       REAL        NOT NULL,
  ssim       REAL        NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS export_runs (
  id         SERIAL PRIMARY KEY,
  user_id    INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  upload_id  INTEGER     NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  format     TEXT        NOT NULL,
  target     TEXT        NOT NULL,
  file_name  TEXT        NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);
