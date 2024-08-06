import urllib.request
from pathlib import Path
import json


# Constants
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

input_file_name = 'input.txt'
dataset_info_file = 'dataset.json'
train_file = 'train.txt'
val_file = 'val.txt'
metadata_file = 'metadata.json'
train_ratio = 0.9
val_ratio = 1.0 - train_ratio


def create_local_path_for_a_file_name(file_name):
    return Path(__file__).parent / file_name


if __name__ == '__main__':
    # Create a local path for each file
    input_file_path = create_local_path_for_a_file_name(input_file_name)
    dataset_info_file_path = create_local_path_for_a_file_name(dataset_info_file)
    train_file_path = create_local_path_for_a_file_name(train_file)
    val_file_path = create_local_path_for_a_file_name(val_file)
    metadata_file_path = create_local_path_for_a_file_name(metadata_file)

    # Download the Tiny Shakespeare dataset
    print(f"Downloading dataset from '{data_url}'...")
    urllib.request.urlretrieve(data_url, input_file_path)

    # Read the dataseat file
    with input_file_path.open(mode='r') as file:
        lines = file.readlines()

    # Show some information about the dataset
    characters = ''.join(lines)
    num_characters = len(characters)
    unique_characters = list(set(characters))
    num_unique_characters = len(unique_characters)
    num_characters_for_training = int(num_characters * train_ratio)
    num_characters_for_validation = num_characters - num_characters_for_training

    print(f"Number of lines: {len(lines)}")
    print(f"Number of characters: {num_characters}")
    print(f"Number of unique characters: {num_unique_characters}")

    print(f"First 30 characters of the file: {repr(characters[:30])}")
    print(f"{num_characters_for_training} characters for training ({train_ratio * 100:.0f}%)")
    print(f"{num_characters_for_validation} characters for validation ({val_ratio * 100:.0f}%)")

    # Create train file
    with train_file_path.open(mode='w') as file:
        file.write(characters[:num_characters_for_training])

    # Create val file
    with val_file_path.open(mode='w') as file:
        file.write(characters[num_characters_for_training:])

    # Create metadata file
    with metadata_file_path.open(mode='w') as file:
        json.dump(
            {
                "unique_chars": unique_characters,
            },
            file,
            indent=2
        )
