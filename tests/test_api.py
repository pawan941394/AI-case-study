from fastapi.testclient import TestClient
from api_For_bot import app

client = TestClient(app)

def test_create_conversation():
    response = client.post(
        "/conversations",
        json={"username": "test_user", "prompt": "Hello"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "response" in data

def test_continue_conversation():
    # First create conversation
    create = client.post(
        "/conversations",
        json={"username": "test_user", "prompt": "Hello"}
    )
    conv_id = create.json()["conversation_id"]

    # Send follow-up
    response = client.post(
        f"/conversations/{conv_id}/messages",
        json={"username": "test_user", "prompt": "Tell me more"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_list_conversations():
    response = client.get("/users/test_user/conversations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
