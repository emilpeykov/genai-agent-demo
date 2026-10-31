from unittest.mock import MagicMock, patch

from app.providers import LLMProviderError, OllamaProvider


def _mock_response(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx

        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
    return mock_resp


@patch("app.providers.httpx.post")
def test_ollama_provider_parses_successful_response(mock_post):
    mock_post.return_value = _mock_response(
        {"response": "Paris is the capital of France.", "eval_count": 8, "prompt_eval_count": 5}
    )

    provider = OllamaProvider(base_url="http://localhost:11434")
    result = provider.generate("What's the capital of France?", "fast")

    assert result.text == "Paris is the capital of France."
    assert result.model == "qwen2.5:7b"  # default OLLAMA_FAST_MODEL
    assert result.estimated_cost_usd == 0.0
    assert result.tokens_used == 13

    called_url = mock_post.call_args[0][0]
    assert called_url == "http://localhost:11434/api/generate"
    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["model"] == "qwen2.5:7b"
    assert called_json["stream"] is False


@patch("app.providers.httpx.post")
def test_ollama_provider_raises_llm_provider_error_when_unreachable(mock_post):
    import httpx

    mock_post.side_effect = httpx.ConnectError("Connection refused")

    provider = OllamaProvider(base_url="http://localhost:11434")
    try:
        provider.generate("hello", "fast")
        assert False, "expected LLMProviderError"
    except LLMProviderError as exc:
        assert "Could not reach Ollama" in str(exc)
