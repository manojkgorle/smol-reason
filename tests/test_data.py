"""Tests for data pipeline."""

from src.data import CANARY_PROBLEMS, CANARY_SOLUTIONS, _make_prompt, get_canary_prompts
from src.utils import extract_boxed, extract_gsm8k_answer


class TestGSM8KAnswerExtraction:
    def test_standard_format(self):
        """Standard GSM8K format with ####."""
        assert extract_gsm8k_answer("blah blah #### 42") == "42"

    def test_with_commas(self):
        """Numbers with commas stripped."""
        assert extract_gsm8k_answer("explanation #### 1,234") == "1234"

    def test_negative(self):
        assert extract_gsm8k_answer("steps #### -5") == "-5"

    def test_decimal(self):
        assert extract_gsm8k_answer("steps #### 3.14") == "3.14"

    def test_no_delimiter(self):
        """Falls back to full string when no #### present."""
        assert extract_gsm8k_answer("just a number 42") == "just a number 42"

    def test_multiline_solution(self):
        """GSM8K solutions often have multi-line reasoning."""
        solution = "Step 1: 10 + 5 = 15\nStep 2: 15 * 2 = 30\n#### 30"
        assert extract_gsm8k_answer(solution) == "30"


class TestBoxedExtraction:
    def test_simple_boxed(self):
        assert extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_boxed(self):
        assert extract_boxed("no boxed here") is None

    def test_deep_nesting(self):
        assert extract_boxed(r"\boxed{\sqrt{x^{2}+1}}") == r"\sqrt{x^{2}+1}"

    def test_multiple_boxed_takes_first(self):
        """When multiple \\boxed, extract the first."""
        text = r"So \boxed{1} or \boxed{2}"
        assert extract_boxed(text) == "1"


class TestPromptFormatting:
    def test_make_prompt_structure(self):
        prompt = _make_prompt("What is 2+2?")
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "What is 2+2?"
        assert "<think>" in prompt[0]["content"]

    def test_canary_prompts(self):
        prompts = get_canary_prompts()
        assert len(prompts) == len(CANARY_PROBLEMS)
        assert len(prompts) == len(CANARY_SOLUTIONS)
        for p in prompts:
            assert len(p) == 2
            assert p[0]["role"] == "system"
            assert p[1]["role"] == "user"
