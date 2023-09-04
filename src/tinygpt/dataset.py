import random
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
    def __getitem__(self, key) -> tuple:
        pass

    @abstractmethod
    def __len__(self) -> tuple:
        pass


class TaskType(Enum):
    NEXT_ELEMENT_PREDICTION = auto()


class NextElementPredictionTask(Task):

    def __init__(self, dataset: Dataset, max_seq_length: int) -> None:
        self.dataset, self.max_seq_length = dataset, max_seq_length
        assert self.dataset is not None, 'dataset is None'
        assert self.max_seq_length is not None, 'max_seq_length is None'
        assert 0 < self.max_seq_length < len(dataset), "0 < max_seq_length < len(dataset)"

        # Iterator index
        self._index = 0

    def __iter__(self) -> type:
        self._index = 0
        return self

    def __next__(self) -> tuple:
        if self._index < len(self):
            input_sequence, expected_output_sequence = self[self._index]
            self._index += 1

            return input_sequence, expected_output_sequence

        else:
            raise StopIteration

    def __getitem__(self, key) -> tuple:
        # input_sequence is a list of max_seq_length characters encoded as numbers taken from the _index position
        input_sequence = [self.dataset.encode(c) for c in self.dataset[key:key + self.max_seq_length]]

        # expected_output_sequence is the same as input_sequence but shifted one character to the right
        expected_output_sequence = [self.dataset.encode(c) for c in self.dataset[key + 1:key + self.max_seq_length + 1]]

        return input_sequence, expected_output_sequence

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


class DatasetHandler:

    def __init__(
        self,
        task: Task,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False
    ) -> None:
        self.task, self.batch_size, self.drop_last, self.shuffle = task, batch_size, drop_last, shuffle

        # Iterator index
        self._index = 0

        # Validate the parameters
        assert task is not None, 'task is None'
        assert 0 < self.batch_size <= len(task), '0 < batch_size <= len(task)'

        # The number of elements to build the batches is defined by the size of the task
        self.num_elements = len(self.task)

        # Calculate the number of batches (ceil division <=> upside-down floor division)
        self.num_batches = -(self.num_elements // -self.batch_size)

        # Check if all batches must have the same number of elements (batch_size elements)
        if drop_last and self.num_elements % self.batch_size:
            self.num_batches -= 1

        # Create the indexex for the task
        self.task_indexes = [i for i in range(self.num_elements)]

    def __iter__(self):
        self._index = 0
        if self.shuffle:
            random.shuffle(self.task_indexes)

        return self

    def __next__(self) -> tuple:
        if self._index < len(self):
            input_batch, expected_output_batch = self[self._index]
            self._index += 1

            return input_batch, expected_output_batch

        else:
            raise StopIteration

    def __getitem__(self, key) -> tuple:
        input_batch = []
        expected_output_batch = []

        start = key * self.batch_size
        stop = min(self.num_elements, start + self.batch_size)
        for idx in self.task_indexes[start:stop]:
            input_sequence, expected_output_sequence = self.task[idx]
            input_batch.append(input_sequence)
            expected_output_batch.append(expected_output_sequence)

        return input_batch, expected_output_batch

    def __len__(self) -> int:
        return self.num_batches
