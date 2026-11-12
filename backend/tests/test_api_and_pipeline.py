import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["app_name"] == "PDF Q&A Assistant"
    assert data["embedding_model"] == settings.EMBEDDING_MODEL_NAME

def test_admin_auth_success_and_failure():
    # 1. Failure with wrong password
    bad_res = client.post("/api/admin/login", json={
        "username": settings.ADMIN_USERNAME,
        "password": "wrong-password"
    })
    assert bad_res.status_code == 401

    # 2. Success with correct credentials
    good_res = client.post("/api/admin/login", json={
        "username": settings.ADMIN_USERNAME,
        "password": settings.ADMIN_PASSWORD
    })
    assert good_res.status_code == 200
    token_data = good_res.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 3. Access protected route with token
    list_res = client.get(
        "/api/admin/pdfs",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert list_res.status_code == 200
    assert "documents" in list_res.json()

    # 4. Access protected route without token -> 401 / 403
    unauth_res = client.get("/api/admin/pdfs")
    assert unauth_res.status_code in (401, 403)

def test_chat_empty_message_validation():
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400
