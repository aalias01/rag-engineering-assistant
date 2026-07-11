"""Regression tests for browser-safe Server-Sent Events framing."""

from api.main import _sse_event


def test_sse_event_prefixes_each_payload_line():
    payload = "Answer.\n[Source: OSHA, Page 46]\nSource text: chlorine"

    assert _sse_event(payload) == (
        "data: Answer.\n"
        "data: [Source: OSHA, Page 46]\n"
        "data: Source text: chlorine\n"
        "\n"
    )


def test_sse_event_keeps_single_line_metadata_intact():
    payload = '__METADATA__:{"route":"factual_lookup"}'

    assert _sse_event(payload) == f"data: {payload}\n\n"
