CREATE DATABASE IF NOT EXISTS cropcare_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE cropcare_ai;

CREATE TABLE IF NOT EXISTS users (
  id CHAR(36) PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  email_confirmed_at DATETIME NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profiles (
  id CHAR(36) PRIMARY KEY,
  full_name VARCHAR(120) NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  farm_name VARCHAR(120) NULL,
  phone VARCHAR(40) NULL,
  location VARCHAR(200) NULL,
  avatar_url VARCHAR(500) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_profiles_user
    FOREIGN KEY (id) REFERENCES users(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS farms (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL UNIQUE,
  name VARCHAR(120) NOT NULL,
  location VARCHAR(200) NULL,
  notes TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_farms_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS plants (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  farm_id CHAR(36) NULL,
  crop_type VARCHAR(80) NOT NULL,
  custom_name VARCHAR(120) NOT NULL,
  zone_or_field VARCHAR(120) NULL,
  planted_at DATE NULL,
  status ENUM('active', 'monitoring', 'diseased', 'archived') NOT NULL DEFAULT 'active',
  notes TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_plants_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_plants_farm
    FOREIGN KEY (farm_id) REFERENCES farms(id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS diagnoses (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  plant_id CHAR(36) NULL,
  image_url VARCHAR(500) NOT NULL,
  image_path VARCHAR(500) NOT NULL,
  predicted_crop VARCHAR(80) NOT NULL,
  health_status ENUM('healthy', 'suspicious', 'diseased') NOT NULL,
  predicted_disease VARCHAR(120) NULL,
  confidence_score DECIMAL(4, 3) NOT NULL,
  severity_level ENUM('low', 'medium', 'high', 'critical') NOT NULL,
  urgency_level ENUM('low', 'medium', 'high', 'urgent') NOT NULL,
  model_version VARCHAR(80) NOT NULL,
  raw_prediction_json JSON NOT NULL,
  ai_advice_text TEXT NOT NULL,
  advisory_payload JSON NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_diagnoses_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_diagnoses_plant
    FOREIGN KEY (plant_id) REFERENCES plants(id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS treatment_logs (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  plant_id CHAR(36) NOT NULL,
  diagnosis_id CHAR(36) NULL,
  treatment_action VARCHAR(200) NOT NULL,
  notes TEXT NULL,
  applied_at DATETIME NOT NULL,
  follow_up_date DATE NULL,
  status ENUM('planned', 'applied', 'monitoring', 'completed') NOT NULL DEFAULT 'planned',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_treatment_logs_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_treatment_logs_plant
    FOREIGN KEY (plant_id) REFERENCES plants(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_treatment_logs_diagnosis
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS notifications (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  type ENUM('diagnosis', 'follow_up', 'treatment', 'system') NOT NULL,
  title VARCHAR(160) NOT NULL,
  message TEXT NOT NULL,
  `read` BOOLEAN NOT NULL DEFAULT FALSE,
  related_diagnosis_id CHAR(36) NULL,
  related_plant_id CHAR(36) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_notifications_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_notifications_diagnosis
    FOREIGN KEY (related_diagnosis_id) REFERENCES diagnoses(id)
    ON DELETE SET NULL,
  CONSTRAINT fk_notifications_plant
    FOREIGN KEY (related_plant_id) REFERENCES plants(id)
    ON DELETE SET NULL
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_profiles_email ON profiles(email);
CREATE INDEX idx_farms_user_id ON farms(user_id);
CREATE INDEX idx_plants_user_id_created_at ON plants(user_id, created_at);
CREATE INDEX idx_plants_user_id_status ON plants(user_id, status);
CREATE INDEX idx_plants_user_id_crop_type ON plants(user_id, crop_type);
CREATE INDEX idx_diagnoses_user_id_created_at ON diagnoses(user_id, created_at);
CREATE INDEX idx_diagnoses_user_id_health_status ON diagnoses(user_id, health_status);
CREATE INDEX idx_diagnoses_plant_id_created_at ON diagnoses(plant_id, created_at);
CREATE INDEX idx_treatment_logs_user_id_applied_at ON treatment_logs(user_id, applied_at);
CREATE INDEX idx_treatment_logs_user_id_follow_up ON treatment_logs(user_id, follow_up_date);
CREATE INDEX idx_notifications_user_id_read_created_at ON notifications(user_id, `read`, created_at);
