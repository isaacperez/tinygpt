# Dataset
## Overview
This [module](../src/tinygpt/dataset.py) provides a framework for handling datasets, with a focus on text data for natural language processing (NLP) tasks. It includes abstract base classes for dataset handling and specific implementations for text-based datasets. Additionally, it offers a handler for batching and iterating over datasets.

### Classes Overview
  - __Dataset__: An abstract base class that defines the necessary methods for any dataset class.
  - __TextDataset__: A concrete implementation of the Dataset class, specifically designed to handle text data.
  - __DatasetHandler__: A utility class that provides batching, shuffling, and iteration functionality for datasets.

## Dataset Class
The Dataset class is an abstract base class (ABC) that defines the interface for any dataset used within TinyGPT. This class must be subclassed, and the following methods must be implemented:
  - `__getitem__`: Retrieves an item from the dataset given an index.
  - `__len__`: Returns the length of the dataset.

### Example Implementation
```python
from typing import Any
from tinygpt.dataset import Dataset


class MyCustomDataset(Dataset):
    def __getitem__(self, idx: int) -> Any:
        # Implementation here
        pass
    
    def __len__(self) -> int:
        # Implementation here
        pass
```

## TextDataset Class
The `TextDataset` class extends the `Dataset` class and is designed for handling text files. It reads text data, tokenizes it using a Byte Pair Encoding (BPE) tokenizer, and prepares the data for training models.

### Constructor
```python
TextDataset(data_file_path: Path, tokenizer: BPETokenizer, max_seq_length: int) -> None
```
  - __`data_file_path`__: The path to the text file containing the dataset.
  - __`tokenizer`__: An instance of `BPETokenizer` to tokenize the text.
  - __`max_seq_length`__: The maximum sequence length for input data.

### Key Features
  - __Tokenization__: Converts the raw text into token IDs using the BPE tokenizer.
  - __Sequence Handling__: Manages sequences of tokens, ensuring that they fit within the specified `max_seq_length`.

### Example Usage

```python
from tinygpt.tokenizer import BPETokenizer
from tinygpt.dataset import TextDataset
from pathlib import Path

tokenizer = BPETokenizer(...)
dataset = TextDataset(Path('data.txt'), tokenizer, max_seq_length=128)
```

### Methods
  - __`__getitem__(self, idx: int) -> tuple`__: Returns a tuple of token IDs for the input and target sequences.
  - __`__len__(self) -> int`__: Returns the number of sequences available in the dataset, adjusted for `max_seq_length`.


## DatasetHandler Class
The `DatasetHandler` class facilitates efficient handling of datasets, providing functionality for batching, shuffling, 
and iterating over the dataset. It is particularly useful when training models on large datasets.

### Constructor
```python
DatasetHandler(dataset: Dataset, batch_size: int, drop_last: bool = False, shuffle: bool = False) -> None
```
  - __`dataset`__: An instance of the `Dataset` class or its subclass.
  - __`batch_size`__: The number of samples per batch.
  - __`drop_last`__: If `True`, drops the last batch if it is smaller than `batch_size`.
  - __`shuffle`__: If `True`, shuffles the dataset before creating batches.

### Key Features
  - __Batching__: Automatically groups data into batches of a specified size.
  - __Shuffling__: Randomly shuffles the dataset at the beginning of each iteration if enabled.
  - __Iteration__: Implements Python’s iterator protocol, allowing for seamless integration into training loops.

### Example Usage
```python
from tinygpt.dataset import DatasetHandler

handler = DatasetHandler(dataset, batch_size=32, shuffle=True)
for batch in handler:
    # Process batch
```

### Methods
  - __`__iter__(self) -> DatasetHandler`__: Resets the iterator and shuffles the dataset if necessary.
  - __`__next__(self) -> tuple`__: Returns the next batch of data.
  - __`__getitem__(self, key: int) -> tuple`__: Retrieves a specific batch based on its index.
  - __`__len__(self) -> int`__: Returns the total number of batches.