from app.core.auth import create_access_token, decode_token, hash_password, verify_password


def test_password_hash_round_trip():
    password = "StrongPass123!"
    stored_hash = hash_password(password)

    assert stored_hash != password
    assert verify_password(password, stored_hash) is True
    assert verify_password("wrong-password", stored_hash) is False


def test_access_token_round_trip():
    token, expires_at = create_access_token(user_id="user-123", email="farmer@example.com")
    payload = decode_token(token)

    assert payload.user_id == "user-123"
    assert payload.email == "farmer@example.com"
    assert payload.token_type == "access"
    assert payload.expires_at == expires_at
