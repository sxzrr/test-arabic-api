from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Text Classification API!"}

def test_predict():
    response = client.post(
        "/predict",
        json={"text": "مدرسة"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
