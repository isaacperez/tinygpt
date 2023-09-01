import re
import json
import pytest

from tinygpt.dataset import DatasetType, Dataset, TextDataset, create_dataset
from tinygpt.dataset import TaskType, Task, NextElementPredictionTask, create_task


def create_data_for_text_dataset(text: str) -> tuple[str, list, dict, dict, dict]:

    data = text
    unique_characters = list(set(data))
    id2char = {i: c for i, c in enumerate(unique_characters)}
    char2id = {c: i for i, c in enumerate(unique_characters)}

    metadata_dict = {
        "unique_chars": unique_characters,
    }

    return data, unique_characters, id2char, char2id, metadata_dict


def test_TextDataset(tmp_path):

    # Test it's subclass of Dataset
    assert issubclass(TextDataset, Dataset)

    # Create a folder for files
    folder_path = tmp_path / 'dataset'
    folder_path.mkdir()

    # Try to create an object with a not valid path
    with pytest.raises(AssertionError, match=".*is not a .txt file"):
        _ = TextDataset(data_file_path=folder_path, metadata_file_path=folder_path, validate_data=False)

    # Create the dataset files
    data_path = folder_path / 'data.txt'
    metadata_path = folder_path / 'metadata.json'

    # Test without creating the files
    with pytest.raises(AssertionError, match=".doesn't exists."):
        _ = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=False)

    # Create the files
    data, unique_characters, _, _, metadata_dict = create_data_for_text_dataset(text="Hello world!")

    with data_path.open(mode='w') as file:
        file.write(data)

    with metadata_path.open(mode='w') as file:
        json.dump(metadata_dict, file)

    # Test with the empty files
    _ = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=False)
    _ = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=True)

    # Create a wrong metadata dict
    metadata_dict["unique_chars"] = metadata_dict["unique_chars"][:-1]
    with metadata_path.open(mode='w') as file:
        json.dump(metadata_dict, file)

    with pytest.raises(AssertionError, match="data has some unknown characters"):
        _ = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=True)

    # Add more characters than are in the text
    metadata_dict["unique_chars"] = unique_characters + ['$']
    with metadata_path.open(mode='w') as file:
        json.dump(metadata_dict, file)

    # Test validate() and has_been_validated() methods
    dataset = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=True)
    assert dataset.has_been_validated()

    dataset = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=False)
    assert not dataset.has_been_validated()
    dataset.validate()
    assert dataset.has_been_validated()

    # Create a valid metadata file for the dataset
    metadata_dict["unique_chars"] = unique_characters
    with metadata_path.open(mode='w') as file:
        json.dump(metadata_dict, file)

    # Create a valid dataset
    dataset = TextDataset(data_file_path=data_path, metadata_file_path=metadata_path, validate_data=True)

    # Test len() method
    assert len(dataset) == len(data)

    # Test get_classes() method
    assert dataset.get_classes() == unique_characters

    # Test getitem() method
    assert all(dataset[i] == c for i, c in enumerate(data))

    # Test encode() and decode() methods
    assert all(dataset.decode(dataset.encode(c)) == c for c in data)


def test_create_dataset(tmp_path):

    # Test not valid dataset types
    for dataset_type in [0, "0"]:
        with pytest.raises(AssertionError, match=f"dataset_type is not a DatasetType. Found {type(dataset_type)}"):
            _ = create_dataset(dataset_type=dataset_type)

    with pytest.raises(AssertionError, match="dataset_type is None"):
        _ = create_dataset(dataset_type=None)

    # Create a folder for files
    folder_path = tmp_path / 'dataset'
    folder_path.mkdir()

    # Create the dataset files
    data_path = folder_path / 'data.txt'
    metadata_path = folder_path / 'metadata.json'

    data, _, _, _, metadata_dict = create_data_for_text_dataset(text="Hello world!")

    with data_path.open(mode='w') as file:
        file.write(data)

    with metadata_path.open(mode='w') as file:
        json.dump(metadata_dict, file)

    # Test dataset has the expected type
    dataset = create_dataset(
        dataset_type=DatasetType.TXT_DATASET,
        data_file_path=data_path,
        metadata_file_path=metadata_path
    )

    assert type(dataset) == TextDataset

    # Test dataset is valid
    dataset.validate()


def test_NextElementPredictionTask(TextDataset_files):

    # Create a dataset from test data
    data_file_path, metadata_file_path = TextDataset_files
    dataset = TextDataset(data_file_path=data_file_path, metadata_file_path=metadata_file_path, validate_data=True)

    # Test it's subclass of Taslk
    assert issubclass(NextElementPredictionTask, Task)

    # Test we can create an object
    _ = NextElementPredictionTask(dataset, max_seq_length=len(dataset) - 1)

    # Try different values for max_seq_length that are not valid
    for wrong_max_seq_length in [-1, 0, len(dataset), len(dataset) + 1]:
        with pytest.raises(AssertionError, match=re.escape("0 < max_seq_length < len(dataset)")):
            _ = NextElementPredictionTask(dataset, max_seq_length=wrong_max_seq_length)

    # Check we can iterate and the data has the expected shape and values
    for max_seq_length in [3, 4, 5]:
        expected_num_iterations = len(dataset) - max_seq_length
        task = NextElementPredictionTask(dataset, max_seq_length=max_seq_length)
        num_iterations = 0
        for (input, expected_output) in task:
            assert len(input) == len(expected_output) == max_seq_length
            assert all(dataset.decode(i) == dataset[num_iterations + idx] for idx, i in enumerate(input))
            assert all(dataset.decode(i) == dataset[num_iterations + idx + 1] for idx, i in enumerate(expected_output))
            num_iterations += 1

        assert num_iterations == expected_num_iterations


def test_create_task(TextDataset_files):

    # Create a dataset from test data
    data_file_path, metadata_file_path = TextDataset_files
    dataset = TextDataset(data_file_path=data_file_path, metadata_file_path=metadata_file_path, validate_data=True)

    # Test not valid dataset types
    for task_type in [0, "0"]:
        with pytest.raises(AssertionError, match=f"task_type is not a TaskType. Found {type(task_type)}"):
            _ = create_task(task_type=task_type)

    with pytest.raises(AssertionError, match="task_type is None"):
        _ = create_task(task_type=None)

    # Test raise error when dataset is not specified
    with pytest.raises(TypeError, match=".*missing 1 required positional argument: 'dataset'"):
        task = create_task(
            task_type=TaskType.NEXT_ELEMENT_PREDICTION,
            max_seq_length=3
        )

    # Test raise error when max_seq_length is not specified
    with pytest.raises(TypeError, match=".*missing 1 required positional argument: 'max_seq_length'"):
        task = create_task(
            task_type=TaskType.NEXT_ELEMENT_PREDICTION,
            dataset=dataset
        )

    # Test task has the expected type
    for max_seq_length in [3, 4, 5]:
        task = create_task(
            task_type=TaskType.NEXT_ELEMENT_PREDICTION,
            dataset=dataset,
            max_seq_length=max_seq_length,
        )

        assert type(task) == NextElementPredictionTask
