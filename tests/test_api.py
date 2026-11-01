from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_tool_path():
    response = client.post("/chat", json={"prompt": "10 * 4"})
    assert response.status_code == 200
    body = response.json()
    assert body["tool_used"] == "calculator"
    assert body["answer"] == "40"


def test_chat_endpoint_llm_path():
    response = client.post("/chat", json={"prompt": "Tell me a fun fact."})
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] in {"fast", "balanced", "powerful"}
    assert "answer" in body


def test_chat_endpoint_rejects_empty_prompt():
    response = client.post("/chat", json={"prompt": ""})
    assert response.status_code == 422


def test_chat_lcel_endpoint_returns_answer():
    response = client.post("/chat/lcel", json={"prompt": "Tell me a fun fact."})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert body["answer"]


def test_chat_lcel_tools_endpoint_returns_answer(monkeypatch):
    # Avoid requiring a real Ollama server in CI -- inject a fake chain with
    # the same Runnable interface the real one would have.
    class _FakeChain:
        def invoke(self, inputs: dict) -> str:
            return f"fake tool-using answer for: {inputs['user_request']}"

    monkeypatch.setattr(main_module, "_lcel_tool_chain", _FakeChain())

    response = client.post("/chat/lcel/tools", json={"prompt": "What is 2 + 2?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake tool-using answer for: What is 2 + 2?"
