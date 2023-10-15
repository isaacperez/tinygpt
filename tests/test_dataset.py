import re
import json
import pytest

from tinygpt.dataset import DatasetType, Dataset, TextDataset, create_dataset
from tinygpt.dataset import TaskType, Task, NextElementPredictionTask, create_task
from tinygpt.dataset import DatasetHandler


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

    assert isinstance(dataset, TextDataset)

    # Test dataset is valid
    dataset.validate()


def test_NextElementPredictionTask(TextDataset_files):
    # Create a dataset from test data
    data_file_path, metadata_file_path = TextDataset_files
    dataset = create_dataset(
        dataset_type=DatasetType.TXT_DATASET,
        data_file_path=data_file_path,
        metadata_file_path=metadata_file_path,
        validate_data=True
    )

    # Test it's subclass of Taslk
    assert issubclass(NextElementPredictionTask, Task)

    # Test we can create an object
    _ = NextElementPredictionTask(dataset=dataset, max_seq_length=len(dataset) - 1)

    # Try different values for max_seq_length that are not valid
    for wrong_max_seq_length in [-1, 0, len(dataset), len(dataset) + 1]:
        with pytest.raises(AssertionError, match=re.escape("0 < max_seq_length < len(dataset)")):
            _ = NextElementPredictionTask(dataset=dataset, max_seq_length=wrong_max_seq_length)

    # Test dataset is not None
    with pytest.raises(AssertionError, match="dataset is None"):
        _ = NextElementPredictionTask(dataset=None, max_seq_length=len(dataset) - 1)

    # Check we can iterate and index the task and the data has the expected shape and values
    for max_seq_length in [3, 4, 5]:
        expected_num_iterations = len(dataset) - max_seq_length
        task = NextElementPredictionTask(dataset=dataset, max_seq_length=max_seq_length)
        num_iterations = 0
        for (input, expected_output) in task:
            assert len(input) == len(expected_output) == max_seq_length
            assert all(dataset.decode(i) == dataset[num_iterations + idx] for idx, i in enumerate(input))
            assert all(dataset.decode(i) == dataset[num_iterations + idx + 1] for idx, i in enumerate(expected_output))
            num_iterations += 1

        assert num_iterations == expected_num_iterations

        for index in [0, len(task) - 1]:
            input, expected_output = task[index]
            assert len(input) == len(expected_output) == max_seq_length
            assert all(dataset.decode(i) == dataset[index + idx] for idx, i in enumerate(input))
            assert all(dataset.decode(i) == dataset[index + idx + 1] for idx, i in enumerate(expected_output))

    # Test that we can iterate more than once over the same dataset
    task = NextElementPredictionTask(dataset=dataset, max_seq_length=max_seq_length)
    for _ in range(3):
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
    dataset = create_dataset(
        dataset_type=DatasetType.TXT_DATASET,
        data_file_path=data_file_path,
        metadata_file_path=metadata_file_path,
        validate_data=True
    )

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

        assert isinstance(task, NextElementPredictionTask)


def test_DatasetHandler(TextDataset_files):
    # Create a dataset from test data
    data_file_path, metadata_file_path = TextDataset_files
    dataset = create_dataset(
        dataset_type=DatasetType.TXT_DATASET,
        data_file_path=data_file_path,
        metadata_file_path=metadata_file_path,
        validate_data=True
    )

    # Create a task
    task = create_task(
        task_type=TaskType.NEXT_ELEMENT_PREDICTION,
        dataset=dataset,
        max_seq_length=3,
    )

    # Test an invalid combination of parameters
    with pytest.raises(AssertionError, match=re.escape("0 < batch_size <= len(task)")):
        for batch_size in [-1, len(task) + 1]:
            dataset_handler = DatasetHandler(
                task=task,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False
            )

    with pytest.raises(AssertionError, match="task is None"):
        dataset_handler = DatasetHandler(
            task=None,
            batch_size=2,
            drop_last=False,
            shuffle=False
        )

    # Test a valid combination of parameters
    num_sequences = len(task)
    for batch_size in range(1, num_sequences + 1):
        for drop_last in [True, False]:
            dataset_handler = DatasetHandler(
                task=task,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=False
            )

            # Check the size is correct
            if drop_last:
                assert len(dataset_handler) == (len(task) // batch_size)
            else:
                assert len(dataset_handler) == (len(task) // batch_size) + int(num_sequences % batch_size > 0)

            # Check batches are correct
            for idx_batch, (input_batch, expected_output_batch) in enumerate(dataset_handler):

                # Check the size of the batch
                assert len(input_batch) == len(expected_output_batch)
                if drop_last or idx_batch < len(dataset_handler) - 1:
                    assert len(input_batch) == batch_size
                else:
                    assert len(input_batch) in {num_sequences % batch_size, batch_size}

                # Check each element of the batch
                for idx_element in range(len(input_batch)):
                    input_seq, expected_output_seq = task[idx_batch * batch_size + idx_element]
                    assert input_seq == input_batch[idx_element]
                    assert expected_output_seq == expected_output_batch[idx_element]

            # Test indexing
            for index in [0, len(dataset_handler) - 1]:
                input_batch, expected_output_batch = dataset_handler[index]

                # Check the size of the batch
                assert len(input_batch) == len(expected_output_batch)
                if drop_last or index < len(dataset_handler) - 1:
                    assert len(input_batch) == batch_size
                else:
                    assert len(input_batch) in {num_sequences % batch_size, batch_size}

                # Check each element of the batch
                for idx_element in range(len(input_batch)):
                    input_seq, expected_output_seq = task[index * batch_size + idx_element]
                    assert input_seq == input_batch[idx_element]
                    assert expected_output_seq == expected_output_batch[idx_element]

    # Test shuffle
    num_sequences = len(task)
    for batch_size in range(1, num_sequences + 1):
        for drop_last in [True, False]:

            # First create one without shuffling to obtain the expected batches
            dataset_handler = DatasetHandler(
                task=task,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=False
            )

            expected_input_sequences = []
            expected_output_sequences = []
            for idx_batch, (input_batch, expected_output_batch) in enumerate(dataset_handler):
                for seq in input_batch:
                    expected_input_sequences.append(seq)
                for seq in expected_output_batch:
                    expected_output_sequences.append(seq)

            # Now create one with shuffling
            dataset_handler = DatasetHandler(
                task=task,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=True
            )

            # Check the size is correct
            if drop_last:
                assert len(dataset_handler) == (len(task) // batch_size)
            else:
                assert len(dataset_handler) == (len(task) // batch_size) + int(num_sequences % batch_size > 0)

            # Check batches are correct
            input_sequences = []
            output_sequences = []
            for idx_batch, (input_batch, expected_output_batch) in enumerate(dataset_handler):

                # Check the size of the batch
                assert len(input_batch) == len(expected_output_batch)
                if drop_last or idx_batch < len(dataset_handler) - 1:
                    assert len(input_batch) == batch_size
                else:
                    assert len(input_batch) in {num_sequences % batch_size, batch_size}

                # Save the sequences
                for seq in input_batch:
                    input_sequences.append(seq)
                for seq in expected_output_batch:
                    output_sequences.append(seq)

            # Check we have the same number of sequences as expected
            assert len(input_sequences) == len(expected_input_sequences)
            assert len(output_sequences) == len(expected_output_sequences)

            # Check that sequences are not repeated and that all expected sequences exist
            idx_sequences_in_expected_input = set()
            for seq in input_sequences:
                if seq in expected_input_sequences:
                    idx_sequences_in_expected_input.add(expected_input_sequences.index(seq))

            idx_sequences_in_expected_output = set()
            for seq in output_sequences:
                if seq in expected_output_sequences:
                    idx_sequences_in_expected_output.add(expected_output_sequences.index(seq))

            num_seq_dropped = num_sequences % batch_size
            if drop_last and num_seq_dropped != 0:
                assert len(idx_sequences_in_expected_input) == len(idx_sequences_in_expected_output)
                assert batch_size - num_seq_dropped <= len(idx_sequences_in_expected_input) < num_sequences
            else:
                assert len(idx_sequences_in_expected_input) == len(expected_input_sequences)
                assert len(idx_sequences_in_expected_output) == len(expected_output_sequences)
