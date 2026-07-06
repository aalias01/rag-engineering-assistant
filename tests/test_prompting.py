from src.generator import REFUSAL_PHRASE, SYSTEM_PROMPT, build_prompt


def test_build_prompt_prefaces_chunks_with_source_and_page():
    prompt = build_prompt(
        "What is a relief valve?",
        [
            {
                "source": "doe_hdbk_1018_v2_mechanical_science.pdf",
                "page": 61,
                "text": "A relief valve opens gradually as pressure rises.",
            }
        ],
    )

    assert "[Excerpt 1" in prompt
    assert "Source: doe_hdbk_1018_v2_mechanical_science.pdf, Page 61" in prompt
    assert "A relief valve opens gradually as pressure rises." in prompt
    assert "Question: What is a relief valve?" in prompt


def test_build_prompt_empty_chunks_uses_no_excerpts_context():
    prompt = build_prompt("What is ASHRAE 62.1?", [])

    assert "(No document excerpts retrieved" in prompt
    assert "answer cannot be grounded" in prompt


def test_refusal_phrase_appears_verbatim_in_system_prompt():
    assert REFUSAL_PHRASE in SYSTEM_PROMPT
