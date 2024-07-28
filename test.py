import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_pdf():
    with open("sample.pdf", "rb") as file:
        response = client.post("/upload_pdf/", files={"file": ("sample.pdf", file, "application/pdf")})
    assert response.status_code == 200
    assert "text" in response.json()

def test_ask():
    query = {
        "query": "What is the capital of France?",
        "context": "Paris is the capital of France."
    }
    response = client.post("/ask/", json=query)
    assert response.status_code == 200
    assert "answer" in response.json()
