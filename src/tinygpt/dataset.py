import math
import random
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod

from tinygpt.tokenizer import BPETokenizer


class Dataset(ABC):

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class TextDataset(Dataset):

    def __init__(self, data_file_path: Path, tokenizer: BPETokenizer, max_seq_length: int) -> None:
        # Save arguments
        self.data_file_path = Path(data_file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Check the arguments
        if self.data_file_path.suffix != ".txt":
            raise ValueError(f"{self.data_file_path} is not a .txt file")
        
        if not self.data_file_path.exists():
            raise RuntimeError(f"File {self.data_file_path} doesn't exists.")

        if not isinstance(tokenizer, BPETokenizer):
            raise RuntimeError(f"Expecting a BPETokenizer, but found {type(tokenizer)}")
        
        if max_seq_length < 1:
            raise ValueError("max_seq_length < 1")

        # Read the data from the file
        self.org_text = self.data_file_path.read_text()

        # Tokenize the text
        self.token_ids = tokenizer.encode(self.org_text, allowed_special="all")

        # Verify max_seq_length is valid
        if self.max_seq_length >= len(self.token_ids):
            raise ValueError(f"max_seq_length >= num. tokens ({len(self.token_ids)})")

    def __getitem__(self, idx: int) -> str:
        # The input starts at idx and ends at idx + max_seq_length, target is the same but with an offset of 1
        return self.token_ids[idx:idx + self.max_seq_length], self.token_ids[idx + 1: idx + self.max_seq_length + 1]

    def __len__(self) -> int:
        return len(self.token_ids) - self.max_seq_length


class DatasetHandler:

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False
    ) -> None:
        
        # Save the arguments
        self.dataset, self.batch_size, self.drop_last, self.shuffle = dataset, batch_size, drop_last, shuffle

        # Iterator index
        self._index = 0

        # Validate the parameters
        if not isinstance(dataset, Dataset):
            raise RuntimeError(f"Expecting a dataset object, but found {type(dataset)}")
        
        if not (0 < batch_size <= len(dataset)):
            raise ValueError("0 < batch_size <= len(dataset)")

        # The number of elements to build the batches is defined by the size of the dataset
        self.num_elements = len(self.dataset)

        # Calculate the number of batches
        self.num_batches = math.ceil(self.num_elements / self.batch_size)

        # Check if all batches must have the same number of elements (batch_size elements)
        if drop_last and self.num_elements % self.batch_size:
            self.num_batches -= 1

        # Create the indexes for the dataset
        self.dataset_indexes = [i for i in range(self.num_elements)]

    def __iter__(self):
        self._index = 0
        if self.shuffle:
            random.shuffle(self.dataset_indexes)

        return self

    def __next__(self) -> tuple:
        if self._index < len(self):
            current_element = self[self._index]
            self._index += 1

            return current_element

        else:
            raise StopIteration

    def __getitem__(self, key: int) -> tuple:
        data = []
        start = key * self.batch_size
        stop = min(self.num_elements, start + self.batch_size)
        for idx in self.dataset_indexes[start:stop]:
            output = self.dataset[idx]
            if isinstance(output, tuple):
                data.append(output)
            else:
                data.append((output,))
        
        # Unpacking the list of tuples and grouping elements by their position
        # The *data unpacks the list of tuples into individual tuples
        # zip(*data) groups the first elements together, the second elements together, and so on
        # map(list, ...) converts each grouped tuple into a list
        packed_data = tuple(map(list, zip(*data)))

        return packed_data

    def __len__(self) -> int:
        return self.num_batches
