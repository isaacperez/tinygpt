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
    else:
        raise NotImplementedError(f'Dataset {dataset_type.name} has not been implemented yet')

    return dataset


class Task(ABC):

    @abstractmethod
    def __iter__(self) -> type:
        pass

    @abstractmethod
    def __next__(self) -> tuple:
        pass

    @abstractmethod
    def __len__(self) -> tuple:
        pass


class TaskType(Enum):
    NEXT_ELEMENT_PREDICTION = auto()


class NextElementPredictionTask(Task):

    def __init__(self, dataset: Dataset, max_seq_length: int) -> None:
        self.dataset, self.max_seq_length = dataset, max_seq_length
        assert self.max_seq_length is not None, 'max_seq_length is None'
        assert 0 < self.max_seq_length < len(dataset), "0 < max_seq_length < len(dataset)"

        self._index = 0

    def __iter__(self) -> type:
        return self

    def __next__(self) -> tuple:

        if self._index < len(self):
            # input_sequence is a list of max_seq_length characters encoded as numbers taken from the _index position
            input_sequence = [
                self.dataset.encode(c) for c in self.dataset[self._index:self._index + self.max_seq_length]
            ]

            # expected_output_sequence is the same as input_sequence but shifted one character to the right
            expected_output_sequence = [
                self.dataset.encode(c) for c in self.dataset[self._index + 1:self._index + self.max_seq_length + 1]
            ]

            self._index += 1

            return input_sequence, expected_output_sequence

        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.dataset) - self.max_seq_length


def create_task(task_type: TaskType, **kwargs) -> Task:

    assert task_type is not None, "task_type is None"
    assert type(task_type) == TaskType, f"task_type is not a TaskType. Found {type(task_type)}"

    if task_type == TaskType.NEXT_ELEMENT_PREDICTION:
        task = NextElementPredictionTask(**kwargs)
    else:
        raise NotImplementedError(f'Task {task_type.name} has not been implemented yet')

    return task
