import re
import ast
import unicodedata
from typing import Optional, Union

from tinygpt.tensor import Tensor


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


def replace_control_characters(s: str) -> str:
    """
    We don't want to print control characters, which distort the output (e.g. \n or much worse)
    https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    http://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape

    return "".join(chars)


def render_token(t: bytes) -> str:
    """Pretty print a token, escaping control characters"""
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)

    return s


class BPETokenizer:

    def __init__(self, regex_pattern: str) -> None:
        self.pattern = regex_pattern
        self.compiled_pattern = re.compile(self.pattern, re.VERBOSE)
        
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

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
            
            # Check if stats is empty
            if not stats:
                print("No more pairs to merge.")
                break

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

    def _encode_bytes(self, text_bytes: bytes, visualise: bool = False) -> list[int]:
        """Convert a stream of bytes into token ids"""
        # Convert all bytes to a list of integers in range 0..255
        ids = list(text_bytes)

        while len(ids) >= 2:
            # See the intermediate merges play out!
            if visualise:
                self._visualise_tokens(ids)

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

    def _encode_without_special_tokens(self, text: str, visualise: bool = False) -> list[int]:
        """Encoding that ignores any special tokens"""
        # Split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        # All chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_bytes(chunk_bytes, visualise)
            ids.extend(chunk_ids)

        return ids

    def encode(self, text: str, allowed_special: Optional[Union[str, set]], visualise: bool = False) -> list[int]:
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
            return self._encode_without_special_tokens(text, visualise)
        
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
                    ids.extend(self._encode_without_special_tokens(part, visualise))

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

    def save(self, file_prefix: str) -> None:
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # Write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # Write the pattern
            f.write(f"{repr(self.pattern)}\n")

            # Write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            # Write the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # Write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # Note: many tokens may be partial utf-8 sequences and cannot be decoded into valid strings. Here we're 
                # using errors='replace' to replace them with the replacement char �. This also means that we couldn't
                # possibly use .vocab in load() because decoding in this way is a lossy operation!
                s = render_token(token)

                # Find the children of this token, if any
                if idx in inverted_merges:
                    # If this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]

                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])

                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")

                else:
                    # Otherwise this is leaf token, just print it (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")
    
    def load(self, model_file: str) -> None:
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")

        # Read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # Read the pattern
            self.pattern = ast.literal_eval(f.readline())
            self.compiled_pattern = re.compile(self.pattern, re.VERBOSE)

            # Read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            # Read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        # Update the internal variables
        self.merges = merges
        self.special_tokens = special_tokens

        # Vocab is simply and deterministically derived from merges and special tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")

    def __call__(
        self, 
        text:Union[str, list[str]], 
        padding_type: str = "none", 
        padding_token: str = "",
        max_length: int = -1, 
        truncation: bool = False,
        return_attention_mask: bool = False,
        visualise: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Processes the input text (either a single string or a list of strings) and returns encoded tokens as tensors. 
        Optionally, it can also return attention masks indicating which tokens are meaningful. The method handles 
        padding, truncation, and attention masks based on the parameters provided.

        Parameters:
        - text: Input text or list of text inputs to be tokenized.
        - padding_type: Specifies the type of padding to be applied. Can be 'none', 'max_length', or 'longest'. 'none' 
          means no padding is applied, 'max_length' pads all sequences to a specific 'max_length', and 'longest' pads 
          all sequences to the length of the longest sequence.
        - padding_token: The token used for padding. This token must be recognized as a special token. 
        - max_length: The maximum length of the sequence after padding and/or truncation. Ignored unless truncate or 
          padding is chosen.
        - truncation: Whether to truncate sequences to 'max_length'.
        - return_attention_mask: If set to True, the method also returns attention masks along with encoded tokens. The 
          attention mask has 1s for real tokens and 0s for padding.
        - visualise: If set to True, it shows how the text is converted into tokens.

        Returns:
        - A tensor of encoded tokens. If 'return_attention_mask' is True, returns a tuple of tensors (encoded tokens, 
          attention masks).
        """

        # Validate the configuration 
        if not (max_length == -1 or max_length > 0):
            raise ValueError(f"max_length is not greater than 0 or -1")
        
        if padding_type != "max_length" and padding_type != "longest" and padding_type != 'none':
            raise ValueError(f"padding_type values are 'max_length', 'longest' or 'none'; but found '{padding_type}'")
        
        if padding_type != "none" and padding_token not in self.special_tokens:
            raise ValueError(f"padding_token ({repr(padding_token)}) is not a special_token")
        
        if padding_type == "max_length" and max_length == -1:
            raise ValueError(f"You need to specify max_length when padding is 'max_length'")
        
        if truncation and max_length == -1:
            raise ValueError(f"You need to specify max_length when truncation is set to True")
        
        # Standarize the text to a List of text
        if isinstance(text, str):
            text = [text]

        # Convert each sentence to tokens 
        sentences_encoded = []
        attention_masks = []
        for sentence in text:
            # Encode the sentence
            sentences_encoded.append(self.encode(sentence, allowed_special="all", visualise=visualise))

            # Save the attention mask
            attention_masks.append([1] * len(sentences_encoded[-1]))

        # Apply padding and truncation
        length_longest_sentence = max(len(sentence) for sentence in sentences_encoded)
        final_sentences_encoded = []
        final_attention_masks = []
        for sentence, attention_mask in zip(sentences_encoded, attention_masks):
            # Apply padding
            if padding_type == "longest":
                num_pad_tokens = length_longest_sentence - len(sentence)
                padding_token_id = self.encode(padding_token, allowed_special="all")
            elif padding_type == "max_length":
                num_pad_tokens = max_length - len(sentence)
                padding_token_id = self.encode(padding_token, allowed_special="all")
            else:
                num_pad_tokens = 0
                padding_token_id = [-1]  # It won't be used

            sentence.extend(padding_token_id * num_pad_tokens)
            attention_mask.extend([0] * num_pad_tokens) 

            # Apply truncation
            if truncation:
                sentence = sentence[:max_length]
                attention_mask = attention_mask[:max_length]
            
            # Save the sentence and attention mask with the padding and truncation applied
            final_sentences_encoded.append(sentence)
            final_attention_masks.append(attention_mask)

            # Check all sequence have the same length
            if len(final_sentences_encoded[0]) != len(final_sentences_encoded[-1]):
                raise RuntimeError("encoded sentences don't have the same length. Use padding or truncation to fix it")

        # Convert the sentences and attention mask into tensors
        final_sentences_encoded = Tensor(final_sentences_encoded)
        final_attention_masks = Tensor(final_attention_masks)

        if return_attention_mask:
            return final_sentences_encoded, final_attention_masks
        else:
            return final_sentences_encoded
        
    def _visualise_tokens(self, token_values: list[int]) -> None:
        """Visualize how the tokenizer has split the sentence into tokens"""
        # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
        # visualise the token. Here, we'll just use the unicode replacement character to represent some
        # fraction of a character.
        unicode_token_values = [self.vocab[x].decode("utf-8", errors="replace") for x in token_values]

        background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]

        running_length = 0
        last_color = None
        for token in unicode_token_values:
            color = background[running_length % len(background)]
            if color == last_color:
                color = background[(running_length + 1) % len(background)]
                assert color != last_color

            last_color = color
            running_length += len(token)

            print(color + token, end="")

        print("\u001b[0m")