from pathlib import Path
import json
from enum import auto, Enum
from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def __getitem__(self, idx) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def encode(self, data) -> int:
        pass

    @abstractmethod
    def decode(self, data) -> str:
        pass

    @abstractmethod
    def get_classes(self) -> list:
        pass

    @abstractmethod
    def has_been_validated(self) -> bool:
        pass

    @abstractmethod
    def validate(self) -> None:
        pass


class DatasetType(Enum):
    TXT_DATASET = auto()


class TextDataset(Dataset):

    def __init__(self, data_file_path: Path, metadata_file_path: Path, validate_data: bool = True) -> None:

        # Save arguments
        self.data_file_path = Path(data_file_path)
        self.metadata_file_path = Path(metadata_file_path)
        self.validate_data = validate_data

        # Check files have valid extension and exist on disk
        assert self.data_file_path.suffix == '.txt', f"{self.data_file_path} is not a .txt file"
        assert self.data_file_path.exists(), f"File {self.data_file_path} doesn't exists."

        assert self.metadata_file_path.suffix == '.json', f"{self.metadata_file_path} is not a .json file"
        assert self.metadata_file_path.exists(), f"File {self.metadata_file_path} doesn't exists."

        # Read the files
        self.data = self.data_file_path.read_text()
        with self.metadata_file_path.open(mode='r') as file:
            self.metadata = json.load(file)

        # Extract relevant information
        self.unique_chars = self.metadata['unique_chars']
        self.char2id = {c: i for i, c in enumerate(self.unique_chars)}
        self.id2char = {i: c for i, c in enumerate(self.unique_chars)}

        # Validate the dataset
        self.dataset_validated = False
        if validate_data:
            self.validate()

    def __getitem__(self, key) -> str:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def encode(self, element) -> int:
        return self.char2id[element]

    def decode(self, element) -> str:
        return self.id2char[element]

    def get_classes(self) -> list:
        return self.unique_chars

    def has_been_validated(self) -> bool:
        return self.dataset_validated

    def validate(self) -> None:
        if not self.has_been_validated():
            assert len(self.data) > 0, 'No data'
            assert len(self.unique_chars) > 0, 'No classes'
            assert len(self.char2id) > 0, 'Empty char2id dict'
            assert len(self.id2char) > 0, 'Empty id2char dict'

            id2char_keys = list(self.id2char.keys())
            id2char_values = list(self.id2char.values())
            char2id_keys = list(self.char2id.keys())
            char2id_values = list(self.char2id.values())

            assert id2char_keys == char2id_values, 'id2char keys != char2id values'
            assert id2char_values == char2id_keys, 'id2char values != char2id keys'
            assert set(char2id_keys) == set(self.unique_chars), 'set(char2id_keys) != set(self.unique_chars)'
            assert all(c in self.unique_chars for c in self.data), 'data has some unknown characters'
            assert all(c == self.id2char[self.char2id[c]] for c in self.data), 'encode->decode does not work'

            self.dataset_validated = True


def create_dataset(dataset_type: DatasetType, **kwargs) -> Dataset:

    assert dataset_type is not None, "dataset_type is None"
    assert type(dataset_type) == DatasetType, f"dataset_type is not a DatasetType. Found {type(dataset_type)}"

    if dataset_type == DatasetType.TXT_DATASET:
        dataset = TextDataset(**kwargs)

    return dataset
