from src.generator import REFUSAL_PHRASE, is_refusal


def test_is_refusal_true_when_phrase_is_present():
    answer = f"{REFUSAL_PHRASE} [Source: none]"

    assert is_refusal(answer) is True


def test_is_refusal_false_without_phrase():
    answer = "A relief valve is used for incompressible fluids."

    assert is_refusal(answer) is False
