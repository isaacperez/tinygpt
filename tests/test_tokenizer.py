import re

from tinygpt.tokenizer import RegexPatterns, get_stats, merge


def test_RegexPatterns():
    # GPT4 pattern
    for seq, expected_tokens in [
        ("", []),
        ("Hello!", ["Hello", "!"]),
        ("I'm We've", ["I", "'m", " We", "'ve"]),
        ("\rHi 2He12llo!", ['\r', 'Hi', ' 2He12llo', '!']),
        ("12345 12 1  -1   ", ["12345", " 12", " 1", " ", " ", "-1", "   "])
    ]:
        assert re.findall(RegexPatterns.GPT4, seq, re.VERBOSE) == expected_tokens


def test_get_stats():
    for ids, expected_output in [
        ([1, 2, 3, 1, 2], {(1, 2): 2, (2, 3): 1, (3, 1): 1}),
        ([2, 2, 2, 2, 2], {(2, 2): 4}),
        ([1, 2, 3, 4, 5], {(1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1}),
    ]:
        # Without updating the counter in place
        assert get_stats(ids) == expected_output

        # Updating the counter in place
        counts = {}
        get_stats(ids, counts=counts)
        assert counts == expected_output


def test_merge():
   assert merge(ids=[1, 1, 2, 2, 1], pair=(1, 2), idx=3) == [1, 3, 2, 1]
   assert merge(ids=[1, 2, 1, 1, 2, 2, 1], pair=(1, 2), idx=3) == [3, 1, 3, 2, 1]
   assert merge(ids=[1, 2, 1, 2, 2, 1, 2], pair=(1, 2), idx=3) == [3, 3, 2, 3]
   assert merge(ids=[1, 2, 1, 2, 2, 1, 2], pair=(4, 5), idx=3) == [1, 2, 1, 2, 2, 1, 2]