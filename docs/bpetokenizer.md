# BPETokenizer
## Overview
The [Tokenizer module](../src/tinygpt/tokenizer.py) provides the Byte Pair Encoding (BPE) tokenizer, which is an essential tool for text processing in natural language processing (NLP) tasks. The BPE tokenizer efficiently encodes text into a sequence of tokens, enabling the model to handle text input effectively.

## Module Overview
  - __`BPETokenizer`__: The main class that implements the Byte Pair Encoding tokenizer, capable of training on text corpora, encoding text into tokens, and decoding tokens back into text.
  - __`RegexPatterns`__: Contains predefined regular expression patterns used by the `BPETokenizer`.
  - __`get_stats`__: A utility function to count consecutive pairs of token IDs.
  - __`merge`__: A function that merges pairs of token IDs during the tokenization process.
  - __`replace_control_characters`__: Replaces control characters in a string to prevent distortion in outputs.
  - __`render_token`__: Pretty prints a token by escaping control character

## BPETokenizer Class
The `BPETokenizer` class implements the Byte Pair Encoding (BPE) algorithm, which is widely used for tokenizing text in NLP tasks. It can be trained on a text corpus to create a vocabulary, encode text into token IDs, and decode token IDs back into text.

### Constructor
```python
BPETokenizer(regex_pattern: str) -> None
```
  - __`regex_pattern`__: A string representing the regular expression pattern used to split text into chunks. This pattern is compiled into a regex object during initialization.

### Key Features
  - __Training__: The tokenizer can be trained on a text corpus to build a vocabulary of token IDs.
  - __Encoding__: Converts text into a list of token IDs using the trained vocabulary and merge rules.
  - __Decoding__: Converts a list of token IDs back into the original text.
  - __Special Tokens__: Supports the registration and handling of special tokens, such as padding or end-of-sequence markers.
  - __Visualization__: Provides a method to visualize how text is tokenized.

### Methods
  - __`train(self, text_corpus: str, vocab_size: int, verbose: bool = False) -> None`__:
    Trains the tokenizer on a given text corpus, generating a vocabulary of the specified size. The verbose flag controls whether progress information is printed.

  - __`encode(self, text: str, allowed_special: Optional[Union[str, set]], visualise: bool = False) -> list[int]`__:
    Encodes a string into a list of token IDs. Handles special tokens based on the allowed_special parameter, which can be `"all"`, `"none"`, `"none_raise"`, or a custom set.

  - __`decode(self, ids: list[int]) -> str`__:
    Decodes a list of token IDs back into a string.

  - __`register_special_tokens(self, special_tokens: dict[str, int]) -> None`__:
    Registers special tokens (e.g., padding, end-of-sequence) with the tokenizer. The `special_tokens` parameter is a dictionary mapping token strings to unique integer IDs.

  - __`save(self, file_prefix: str) -> None`__:
    Saves the tokenizer's model and vocabulary to files. The model is saved in a format suitable for loading later, while the vocabulary is saved in a human-readable format.

  - __`load(self, model_file: str) -> None`__:
    Loads a tokenizer model from a file, reconstructing the vocabulary and merge rules.

  - __`__call__(self, text: Union[str, list[str]], padding_type: str = "none", padding_token: str = "", max_length: int = -1, truncation: bool = False, return_attention_mask: bool = False, visualise: bool = False) -> Union[Tensor, tuple[Tensor, Tensor]]`__:
    Tokenizes the input text (or a list of texts) and returns encoded tokens as tensors. Supports padding, truncation, and attention masks.

  - __`_visualise_tokens(self, token_values: list[int]) -> None`__:
    Visualizes how the tokenizer has split the text into tokens. This is useful for understanding how the tokenizer processes text.

### Example Usage
```python
from tinygpt.tokenizer import BPETokenizer, RegexPatterns

# Initialize the tokenizer with a predefined regex pattern
tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)

# Train the tokenizer on a text corpus
text_corpus = "This is a simple text corpus for training the tokenizer."
tokenizer.train(text_corpus, vocab_size=10000)

# Encode a sentence
encoded_tokens = tokenizer.encode("This is a test sentence.", allowed_special="all")
print("Encoded Tokens:", encoded_tokens)

# Decode the tokens back to text
decoded_text = tokenizer.decode(encoded_tokens)
print("Decoded Text:", decoded_text)

# Save the tokenizer model
tokenizer.save("bpe_tokenizer")

# Load the tokenizer model
tokenizer.load("bpe_tokenizer.model")
```