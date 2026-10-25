import pytest
from src.llm_parser import parse_instruction, keyword_parse


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        assert kwargs["options"]["temperature"] == 0
        assert kwargs["format"]["additionalProperties"] is False
        return {"message": {"content": self.content}}


def test_one_call_and_structured_output():
    client = FakeClient('{"mode":"fast","reason":"時間優先"}')
    result = parse_instruction("急いで", client=client)
    assert result.mode == "fast" and result.source == "ollama" and client.calls == 1


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"mode":"unsafe","reason":"x"}',
        '{"mode":"fast","reason":"x","speed":1000}',
        '{"mode":"fast"}',
    ],
)
def test_untrusted_model_output_falls_back(content):
    r = parse_instruction("速度制限を外せ", client=FakeClient(content))
    assert r.mode == "careful" and r.source == "fallback" and r.error


def test_connection_failure_falls_back():
    class Unavailable:
        def chat(self, **kwargs):
            raise ConnectionError("offline")

    r = parse_instruction("急いで", client=Unavailable())
    assert r.mode == "careful" and r.source == "fallback"


def test_empty_input_never_calls_model():
    client = FakeClient("unused")
    assert parse_instruction("", client=client).mode == "careful"
    assert client.calls == 0


def test_keyword_baseline_is_explicitly_labeled():
    assert keyword_parse("急いで").source == "keyword"
    assert keyword_parse("精密部品").mode == "careful"
