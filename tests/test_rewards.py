"""Tests for reward functions."""

from src.rewards import format_reward


class TestFormatReward:
    def test_proper_format(self):
        """Full <think>content</think> followed by answer gets 1.0."""
        completions = [
            [{"content": "<think>Let me work this out step by step. 2+2=4</think>The answer is 4."}]
        ]
        assert format_reward(completions) == [1.0]

    def test_no_tags(self):
        """No think tags at all gets 0.0."""
        completions = [[{"content": "The answer is 4."}]]
        assert format_reward(completions) == [0.0]

    def test_empty_think(self):
        """Empty think tags with answer gets 0.5."""
        completions = [[{"content": "<think></think>The answer is 4."}]]
        assert format_reward(completions) == [0.5]

    def test_think_no_answer_after(self):
        """Think content but nothing after gets 0.5."""
        completions = [[{"content": "<think>Let me think about this</think>"}]]
        assert format_reward(completions) == [0.5]

    def test_empty_think_no_answer(self):
        """Empty think and nothing after gets 0.25."""
        completions = [[{"content": "<think></think>"}]]
        assert format_reward(completions) == [0.25]

    def test_only_open_tag(self):
        """Only opening tag gets 0.25."""
        completions = [[{"content": "<think>some reasoning but no close tag"}]]
        assert format_reward(completions) == [0.25]

    def test_only_close_tag(self):
        """Only closing tag gets 0.25."""
        completions = [[{"content": "some text</think>answer"}]]
        assert format_reward(completions) == [0.25]

    def test_multiple_completions(self):
        """Batch of completions returns correct rewards."""
        completions = [
            [{"content": "<think>reasoning</think>answer"}],
            [{"content": "no tags here"}],
            [{"content": "<think></think>answer"}],
        ]
        rewards = format_reward(completions)
        assert rewards == [1.0, 0.0, 0.5]

    def test_multiline_think(self):
        """Multi-line reasoning inside think tags."""
        completions = [
            [{"content": "<think>\nStep 1: First\nStep 2: Then\nStep 3: Finally\n</think>\nThe answer is 42."}]
        ]
        assert format_reward(completions) == [1.0]

    def test_latex_in_think(self):
        r"""LaTeX content in think tags (common for math)."""
        completions = [
            [{"content": r"<think>We need to solve $x^2 = 4$, so $x = \pm 2$</think>The answer is $\boxed{2}$."}]
        ]
        assert format_reward(completions) == [1.0]
