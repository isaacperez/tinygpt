import re

import pytest 

from tinygpt.tokenizer import RegexPatterns, get_stats, merge, BPETokenizer


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


def test_train():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)

    # Before training, a tokenizer doesn't have vocabulary nor merges    
    assert tokenizer.vocab == {}
    assert tokenizer.merges == {}

    # Train with some examples
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)
    
    # Validate the vocabulary and new merges
    assert len(tokenizer.vocab) == 257
    assert len(tokenizer.merges) == 1
    
    assert tokenizer.vocab[256] == "12".encode("utf-8")
    assert tokenizer.merges[(49, 50)] == 256


def test_encode_bytes():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)

    assert tokenizer._encode_bytes("12".encode("utf-8")) == [256]


def test_register_special_tokens():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)
    with pytest.raises(ValueError):
        tokenizer.register_special_tokens({"HELLO": 1, "HI": 2})
    
    tokenizer.register_special_tokens({"HELLO": 258, "HI": 259})
    tokenizer.register_special_tokens({"A": 260, "B": 261})

    assert tokenizer.special_tokens == {"A": 260, "B": 261}
    assert tokenizer.inverse_special_tokens == {260: "A", 261: "B"}


def test_encode_without_special_tokens():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)

    assert tokenizer._encode_without_special_tokens("12123") == [256, 256, 51]


def test_encode():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)
    
    tokenizer.register_special_tokens({"A": 260, "B": 261})

    # Without special tokens
    assert tokenizer.encode(text="123", allowed_special="all") == [256, 51]
    assert tokenizer.encode(text="123", allowed_special="none") == [256, 51]
    assert tokenizer.encode(text="123", allowed_special="none_raise") == [256, 51]
    assert tokenizer.encode(text="123", allowed_special=set("A")) == [256, 51]

    # With special tokens
    assert tokenizer.encode(text="123AB", allowed_special="all") == [256, 51, 260, 261]
    assert tokenizer.encode(text="123AB", allowed_special="none") == [256, 51, 65, 66]
    
    with pytest.raises(AssertionError):
        tokenizer.encode(text="123AB", allowed_special="none_raise")

    assert tokenizer.encode(text="123AB", allowed_special=set("A")) == [256, 51, 260, 66]


def test_decode():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)

    assert tokenizer.decode(ids=[]) == ""
    with pytest.raises(ValueError):
        tokenizer.decode(ids=[12345])

    assert tokenizer.decode(ids=[65, 66, 256]) == "AB12"