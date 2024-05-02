import re
from typing import Optional, Union


class RegexPatterns:
    GPT4 = r"""
        # This is an approximation of the pattern used by the tokenizer of GPT-4 but using re library instead of regex
        (?i:[sdmt]|ll|ve|re)| # Contractions, case insensitive
        [^\r\n\w\d]?+\w+|     # Words, allowing an optional non-alphabetic/non-numeric character at the start
        [^\s\w\d]++[\r\n]*|   # Sequences of non-letters, non-digits, and non-spaces, possibly followed by new lines
        \s*[\r\n]|            # Spaces followed by new lines
        \s+(?!\S)|            # Spaces at the end of a line
        \s+                   # One or more spaces
        """


def get_stats(ids: list[int], counts: dict[tuple[int, int], int] = None) -> dict[tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    In the list of integers (ids), replace all consecutive occurrences of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids


class BPETokenizer:

    def __init__(self, regex_pattern: str) -> None:
        self.pattern = regex_pattern
        self.compiled_pattern = re.compile(self.pattern, re.VERBOSE)
        
        self.merges = {}
        self.vocab = {}

        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text_corpus: str, vocab_size: int, verbose: bool = False) -> None:
        # Vocabulary always includes all bytes
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text_corpus)

        # Preprocessing: we encode the text in bytes using utf-8 and store the bytes as a list of integers
        ids = [list(char.encode("utf-8")) for char in text_chunks]

        # Iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # Count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # Passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)

            # Mint a new token: assign it the next available id
            idx = 256 + i

            # Replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # Save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        """Convert a stream of bytes into token ids"""
        # Convert all bytes to integers in range 0..255
        ids = list(text_bytes)

        while len(ids) >= 2:
            # Find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # If there are no more merges available, the key will result in an inf for every single pair, and the min 
            # will be just the first pair in the list, arbitrarily we can detect this terminating case by a membership 
            # check
            if pair not in self.merges:
                break  # nothing else can be merged anymore

            # Otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def _encode_without_special_tokens(self, text: str) -> list[int]:
        """Encoding that ignores any special tokens"""
        # Split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        # All chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_bytes(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    def encode(self, text: str, allowed_special: Optional[Union[str, set]]) -> list[int]:
        """
        Encode a string into a list of integers. Unlike _encode, this function handles special tokens.
        `allowed_special` can be "all"|"none"|"none_raise" or a custom set of special tokens. if `none_raise`, then an 
        error is raised if any special token is encountered in text this is the default tiktoken behavior right now as 
        well any other behavior is either annoying, or a major footgun.
        """
        # Prepare special configuration
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # Shortcut: if no special tokens, just use the ordinary encoding
            return self._encode_without_special_tokens(text)
        
        else:
            # We have to be careful with potential special tokens in text. We handle special tokens by splitting the 
            # text based on the occurrence of any exact match with any of the special tokens.
            # Note that surrounding the pattern with () makes it into a capturing group, so the special tokens will be 
            # included.
            special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
            special_chunks = re.split(special_pattern, text)

            # Now all the special characters are separated from the rest of the text. All chunks of text are encoded 
            # separately, then results are joined
            ids = []
            for part in special_chunks:
                if part in special:
                    # This is a special token, encode it separately as a special case
                    ids.append(special[part])
                else:
                    # This is an ordinary sequence, encode it normally
                    ids.extend(self._encode_without_special_tokens(part))

            return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integers into a string"""
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text
    
    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """special_tokens is a dictionary of str -> int. Example: {"<|endoftext|>": 100257}"""
        # Check new ids are not already in use in the vocabulary
        vocabulary_ids = set(self.vocab.keys())
        for special_token, token_id in special_tokens.items():
            # Validate types
            assert isinstance(special_token, str)
            assert isinstance(token_id, int)

            # Check current id is new
            if token_id in vocabulary_ids:
                raise ValueError("special_tokens use ids that are already in use")

        # Print a warning if we already have special_tokens
        if len(self.special_tokens) > 0:
            print("Warning. This tokenizer already has special tokens. Updating it with the new ones...")

        # Save the special tokens
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}