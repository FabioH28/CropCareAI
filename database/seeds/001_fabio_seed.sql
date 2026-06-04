USE cropcare_ai;

INSERT INTO users (
  id,
  email,
  password_hash,
  email_confirmed_at,
  is_active
)
VALUES (
  'ca122bc0-4d81-4c4c-ac0c-8927e27e6b9c',
  'fabio@cropcareai.app',
  'pbkdf2_sha256$390000$01gHiwx3_jNze8dkQu6tNw==$symZVjh7ue8tKFb9qksACX71pmgdPGyrH4XTCf_dn_Y=',
  CURRENT_TIMESTAMP,
  TRUE
)
ON DUPLICATE KEY UPDATE
  email = VALUES(email),
  password_hash = VALUES(password_hash),
  email_confirmed_at = VALUES(email_confirmed_at),
  is_active = VALUES(is_active);

INSERT INTO profiles (
  id,
  full_name,
  email,
  farm_name,
  phone
)
VALUES (
  'ca122bc0-4d81-4c4c-ac0c-8927e27e6b9c',
  'Fabio',
  'fabio@cropcareai.app',
  'Fabio Farm',
  NULL
)
ON DUPLICATE KEY UPDATE
  full_name = VALUES(full_name),
  email = VALUES(email),
  farm_name = VALUES(farm_name),
  phone = VALUES(phone);
