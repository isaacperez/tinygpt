import re
import os 

import pytest 

from tinygpt.tensor import Tensor
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
    assert len(tokenizer.vocab) == 256
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


def test_encode_decode():
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)

    assert tokenizer.decode(ids=[]) == ""
    assert tokenizer.decode(ids=tokenizer.encode("123412561278", allowed_special="none")) == "123412561278"


def test_save_and_load(tmp_path):
    # Create a temp folder
    tmp_dir = tmp_path / "test_tokenizer"
    tmp_dir.mkdir()
    
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)

    model_path = str(tmp_dir / "gpt4_tokenizer")
    tokenizer.save(model_path)

    assert os.path.isfile(model_path + ".model")
    assert os.path.isfile(model_path + ".vocab")

    new_tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    new_tokenizer.load(model_path + ".model")

    assert tokenizer.pattern == new_tokenizer.pattern
    assert tokenizer.compiled_pattern == new_tokenizer.compiled_pattern
    assert tokenizer.vocab == new_tokenizer.vocab
    assert tokenizer.merges == new_tokenizer.merges
    assert tokenizer.special_tokens == new_tokenizer.special_tokens
    assert tokenizer.inverse_special_tokens == new_tokenizer.inverse_special_tokens


def test_call():
    # Create a tokenizer with a special token for padding
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus="123412561278", vocab_size=257, verbose=False)
    tokenizer.register_special_tokens({"[PAD]": 258})

    # Default config
    input_ids = tokenizer(text="12123[PAD]")

    assert isinstance(input_ids, Tensor)
    assert input_ids.shape == (1, 4)
    assert input_ids.to_python() == [[256, 256, 51, 258]]

    # Attention mask
    input_ids, attention_masks = tokenizer(text="12123[PAD]", return_attention_mask=True)

    assert isinstance(input_ids, Tensor)
    assert input_ids.shape == (1, 4)
    assert input_ids.to_python() == [[256, 256, 51, 258]]

    assert isinstance(attention_masks, Tensor)
    assert attention_masks.shape == (1, 4)
    assert attention_masks.to_python() == [[1, 1, 1, 1]]

    # More than one text
    input_ids, attention_masks = tokenizer(text=["12123[PAD]", "4564"], return_attention_mask=True)

    assert isinstance(input_ids, Tensor)
    assert input_ids.shape == (2, 4)
    assert input_ids.to_python() == [[256, 256, 51, 258], [52, 53, 54, 52]]

    assert isinstance(attention_masks, Tensor)
    assert attention_masks.shape == (2, 4)
    assert attention_masks.to_python() == [[1, 1, 1, 1], [1, 1, 1, 1]]

    # Raise an error when they have different length
    with pytest.raises(RuntimeError):
        input_ids, attention_masks = tokenizer(text=["12123[PAD]", "456"], return_attention_mask=True)

    # Padding ('longest')
    input_ids, attention_masks = tokenizer(
        text=["1212", "456"], 
        padding_type="longest",
        padding_token="[PAD]",
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[256, 256, 258], [52, 53, 54]]
    assert attention_masks.to_python() == [[1, 1, 0], [1, 1, 1]]

    input_ids, attention_masks = tokenizer(
        text=["456", "1212"], 
        padding_type="longest",
        padding_token="[PAD]",
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[52, 53, 54], [256, 256, 258]]
    assert attention_masks.to_python() == [[1, 1, 1], [1, 1, 0]]

    # Padding ('max_length')
    with pytest.raises(ValueError):
        input_ids, attention_masks = tokenizer(
            text=["1212", "456"], 
            padding_type="max_length",
            padding_token="[PAD]",
            max_length=-1,
            return_attention_mask=True
        )

    input_ids, attention_masks = tokenizer(
        text=["1212", "456"], 
        padding_type="max_length",
        padding_token="[PAD]",
        max_length=4,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[256, 256, 258, 258], [52, 53, 54, 258]]
    assert attention_masks.to_python() == [[1, 1, 0, 0], [1, 1, 1, 0]]

    input_ids, attention_masks = tokenizer(
        text=["456", "1212"], 
        padding_type="max_length",
        padding_token="[PAD]",
        max_length=4,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[52, 53, 54, 258], [256, 256, 258, 258]]
    assert attention_masks.to_python() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    
    with pytest.raises(RuntimeError):
        # It doesn't pad the text but they have different lengths
        input_ids, attention_masks = tokenizer(
            text=["456", "1212"], 
            padding_type="max_length",
            padding_token="[PAD]",
            max_length=2,
            return_attention_mask=True
        )

    # Truncation
    with pytest.raises(ValueError):
        input_ids, attention_masks = tokenizer(
            text=["1212", "456"], 
            truncation=True,
            max_length=-1,
            return_attention_mask=True
        )
        
    input_ids, attention_masks = tokenizer(
        text=["1212", "456"], 
        truncation=True,
        max_length=2,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[256, 256], [52, 53]]
    assert attention_masks.to_python() == [[1, 1], [1, 1]]

    input_ids, attention_masks = tokenizer(
        text=["456", "1212"], 
        truncation=True,
        max_length=2,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[52, 53], [256, 256]]
    assert attention_masks.to_python() == [[1, 1], [1, 1]]

    # max_length longer than the sequences' length
    input_ids, attention_masks = tokenizer(
        text=["456", "121212"], 
        truncation=True,
        max_length=5,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[52, 53, 54], [256, 256, 256]]
    assert attention_masks.to_python() == [[1, 1, 1], [1, 1, 1]]

    with pytest.raises(RuntimeError):
        # It doesn't truncate the text but they have different lengths
        input_ids, attention_masks = tokenizer(
            text=["456", "1212"], 
            truncation=True,
            max_length=5,
            return_attention_mask=True
        )

    # Padding and truncation
    input_ids, attention_masks = tokenizer(
        text=["12121212", "45"], 
        padding_type="longest",
        padding_token="[PAD]",
        max_length=3,
        truncation=True,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[256, 256, 256], [52, 53, 258]]
    assert attention_masks.to_python() == [[1, 1, 1], [1, 1, 0]]

    input_ids, attention_masks = tokenizer(
        text=["45", "12121212"], 
        padding_type="longest",
        padding_token="[PAD]",
        max_length=3,
        truncation=True,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[52, 53, 258], [256, 256, 256]]
    assert attention_masks.to_python() == [[1, 1, 0], [1, 1, 1]]

    input_ids, attention_masks = tokenizer(
        text=["45", "12121212"], 
        padding_type="longest",
        padding_token="[PAD]",
        max_length=4,
        truncation=True,
        return_attention_mask=True
    )

    assert input_ids.to_python() == [[52, 53, 258, 258], [256, 256, 256, 256]]
    assert attention_masks.to_python() == [[1, 1, 0, 0], [1, 1, 1, 1]]
