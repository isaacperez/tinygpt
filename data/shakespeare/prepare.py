import requests
import os
import json


def create_local_path_for_a_file_name(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


# Constants
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

input_file_name = 'input.txt'
input_file_path = create_local_path_for_a_file_name(input_file_name)

dataset_info_file = 'dataset.json'
dataset_info_file_path = create_local_path_for_a_file_name(dataset_info_file)

train_file = 'train.txt'
train_file_path = create_local_path_for_a_file_name(train_file)

val_file = 'val.txt'
val_file_path = create_local_path_for_a_file_name(val_file)

metadata_file = 'metadata.json'
metadata_file_path = create_local_path_for_a_file_name(metadata_file)

train_ratio = 0.8
val_ratio = 1.0 - train_ratio


if __name__ == '__main__':

    # Download the Tiny Shakespeare dataset
    print(f"Downloading dataset from '{data_url}'...")
    with open(input_file_path, mode='w') as file:
        file.write(requests.get(data_url).text)

    # Read the dataseat file
    with open(input_file_path, mode='r') as file:
        lines = file.readlines()

    # Show some information about the dataset
    characters = ''.join(lines)
    num_characters = len(characters)
    unique_characters = list(set(characters))
    num_unique_characters = len(unique_characters)
    num_characters_for_training = int(num_characters * train_ratio)
    num_characters_for_validation = int(num_characters * val_ratio)

    print(f"Number of lines: {len(lines)}")
    print(f"Number of characters: {num_characters}")
    print(f"Number of unique characters: {num_unique_characters}")

    print(f"First 30 characters of the file: {repr(characters[:30])}")
    print(f"{num_characters_for_training} characters for training ({train_ratio * 100:.0f}%)")
    print(f"{num_characters_for_validation} characters for validation ({val_ratio * 100:.0f}%)")

    # Create train file
    with open(train_file_path, mode='w') as file:
        file.write(characters[:int(num_characters * train_ratio)])

    # Create val file
    with open(val_file_path, mode='w') as file:
        file.write(characters[int(num_characters * train_ratio):])

    # Create metadata file
    with open(metadata_file, mode='w') as file:
        json.dump(
            {
                "unique_chars": unique_characters,
            },
            file,
            indent=2
        )
