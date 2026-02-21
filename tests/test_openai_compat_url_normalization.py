from openai_compat import OpenAICompatibleChatClient


def test_normalize_urls_accepts_root_and_v1() -> None:
    root, v1 = OpenAICompatibleChatClient._normalize_urls("http://127.0.0.1:1245")
    assert root == "http://127.0.0.1:1245"
    assert v1 == "http://127.0.0.1:1245/v1"

    root, v1 = OpenAICompatibleChatClient._normalize_urls("http://127.0.0.1:1245/v1")
    assert root == "http://127.0.0.1:1245"
    assert v1 == "http://127.0.0.1:1245/v1"


def test_normalize_urls_trims_full_endpoint_paths() -> None:
    root, v1 = OpenAICompatibleChatClient._normalize_urls("http://127.0.0.1:1245/v1/chat/completions")
    assert root == "http://127.0.0.1:1245"
    assert v1 == "http://127.0.0.1:1245/v1"

    root, v1 = OpenAICompatibleChatClient._normalize_urls("127.0.0.1:1245/v1/models")
    assert root == "http://127.0.0.1:1245"
    assert v1 == "http://127.0.0.1:1245/v1"


def test_normalize_urls_keeps_api_prefix() -> None:
    root, v1 = OpenAICompatibleChatClient._normalize_urls("http://localhost:8080/api/v1/chat/completions")
    assert root == "http://localhost:8080/api"
    assert v1 == "http://localhost:8080/api/v1"


def test_normalize_urls_does_not_trim_v1beta() -> None:
    root, v1 = OpenAICompatibleChatClient._normalize_urls("http://localhost:8080/v1beta")
    assert root == "http://localhost:8080/v1beta"
    assert v1 == "http://localhost:8080/v1beta/v1"

