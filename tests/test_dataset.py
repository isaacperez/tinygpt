import json
import pytest

from tinygpt.dataset import DatasetType, Dataset, TextDataset, create_dataset


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

    # Test dataset have the expected type
    dataset = create_dataset(
        dataset_type=DatasetType.TXT_DATASET,
        data_file_path=data_path,
        metadata_file_path=metadata_path
    )

    assert type(dataset) == TextDataset

    # Test dataset is valid
    dataset.validate()
