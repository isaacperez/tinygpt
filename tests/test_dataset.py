import pytest

from tinygpt.dataset import TextDataset, DatasetHandler
from tinygpt.tokenizer import BPETokenizer, RegexPatterns


def test_TextDataset(tmp_path):
    # Create a folder for files
    folder_path = tmp_path / 'dataset'
    folder_path.mkdir()
    text_data = "Hello world! How are you?"

    # Create a tokenizer with a special token for padding
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus=text_data, vocab_size=512, verbose=False)

    # Try to create an object with wrong argument
    with pytest.raises(ValueError):
        _ = TextDataset(data_file_path=folder_path, tokenizer=tokenizer, max_seq_length=2)
    
    # Create the dataset file
    data_path = folder_path / 'data.txt'

    # Try with a file that doesn't exist
    with pytest.raises(RuntimeError):
        _ = TextDataset(data_file_path=data_path, tokenizer=tokenizer, max_seq_length=2)   

    # Populate the file with the text data
    with data_path.open(mode='w') as file:
        file.write(text_data)

    # Try a wrong value for max_seq_length
    with pytest.raises(ValueError):
        _ = TextDataset(data_file_path=folder_path, tokenizer=tokenizer, max_seq_length=1)

    with pytest.raises(RuntimeError):
        _ = TextDataset(data_file_path=data_path, tokenizer=None, max_seq_length=2)   
    
    with pytest.raises(ValueError):
        _ = TextDataset(data_file_path=data_path, tokenizer=tokenizer, max_seq_length=7)  

    # Create a valid dataset
    dataset = TextDataset(data_file_path=data_path, tokenizer=tokenizer, max_seq_length=3)
    
    # Test len() method
    assert len(dataset) == 4

    # Test getitem() method
    expected_inputs_and_targets = [
        ("Hello world!", " world! How"),
        (" world! How", "! How are"),
        ("! How are",  " How are you"),
        (" How are you", " are you?"),
    ]
    for i in range(len(dataset)):
        input_ids, target_ids = dataset[i]
        expected_input_ids, expected_target_ids = expected_inputs_and_targets[i]

        assert input_ids == tokenizer.encode(expected_input_ids, allowed_special="all")
        assert target_ids == tokenizer.encode(expected_target_ids, allowed_special="all")

    # Max seq length
    dataset = TextDataset(data_file_path=data_path, tokenizer=tokenizer, max_seq_length=6)
    input_ids, target_ids = dataset[0]

    assert tokenizer.decode(input_ids) == 'Hello world! How are you'
    assert tokenizer.decode(target_ids) == ' world! How are you?'


def test_DatasetHandler(tmp_path):
    # Create a folder for files
    folder_path = tmp_path / 'dataset'
    folder_path.mkdir()

    # Create the data file
    data_path = folder_path / 'data.txt'
    text_data = "Hello world! How are you?"
    with data_path.open(mode='w') as file:
        file.write(text_data)

    # Create a tokenizer with a special token for padding
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)
    tokenizer.train(text_corpus=text_data, vocab_size=512, verbose=False)

    # Create a dataset from test data
    dataset = TextDataset(data_file_path=data_path, tokenizer=tokenizer, max_seq_length=3)

    # Test an invalid combination of parameters
    with pytest.raises(ValueError):
        for batch_size in [-1, len(dataset) + 1]:
            dataset_handler = DatasetHandler(dataset=dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    with pytest.raises(RuntimeError):
        dataset_handler = DatasetHandler(dataset=None, batch_size=2, drop_last=False, shuffle=False)

    # Test a valid combination of parameters
    num_sequences = len(dataset)
    for batch_size in range(1, num_sequences + 1):
        for drop_last in [True, False]:
            dataset_handler = DatasetHandler(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)

            # Check the size is correct
            if drop_last:
                assert len(dataset_handler) == (len(dataset) // batch_size)
            else:
                assert len(dataset_handler) == (len(dataset) // batch_size) + int(num_sequences % batch_size > 0)

            # Check batches are correct
            for idx_batch, (input_batch, target_batch) in enumerate(dataset_handler):

                # Check the size of the batch
                assert len(input_batch) == len(target_batch)
                if drop_last or idx_batch < len(dataset_handler) - 1:
                    assert len(input_batch) == batch_size
                else:
                    assert len(input_batch) in {num_sequences % batch_size, batch_size}

                # Check each element of the batch
                for idx_element in range(len(input_batch)):
                    input_seq, target_seq = dataset[idx_batch * batch_size + idx_element]
                    assert input_seq == input_batch[idx_element]
                    assert target_seq == target_batch[idx_element]

            # Test indexing
            for index in [0, len(dataset_handler) - 1]:
                input_batch, target_batch = dataset_handler[index]

                # Check the size of the batch
                assert len(input_batch) == len(target_batch)
                if drop_last or index < len(dataset_handler) - 1:
                    assert len(input_batch) == batch_size
                else:
                    assert len(input_batch) in {num_sequences % batch_size, batch_size}

                # Check each element of the batch
                for idx_element in range(len(input_batch)):
                    input_seq, target_seq = dataset[index * batch_size + idx_element]
                    assert input_seq == input_batch[idx_element]
                    assert target_seq == target_batch[idx_element]

    # Test shuffle
    num_sequences = len(dataset)
    for batch_size in range(1, num_sequences + 1):
        for drop_last in [True, False]:

            # First create one without shuffling to obtain the expected batches
            dataset_handler = DatasetHandler(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)

            expected_input_sequences = []
            expected_target_sequences = []
            for idx_batch, (input_batch, target_batch) in enumerate(dataset_handler):
                for seq in input_batch:
                    expected_input_sequences.append(seq)
                for seq in target_batch:
                    expected_target_sequences.append(seq)

            # Now create one with shuffling
            dataset_handler = DatasetHandler(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)

            # Check the size is correct
            if drop_last:
                assert len(dataset_handler) == (len(dataset) // batch_size)
            else:
                assert len(dataset_handler) == (len(dataset) // batch_size) + int(num_sequences % batch_size > 0)

            # Check batches are correct
            input_sequences = []
            target_sequences = []
            for idx_batch, (input_batch, target_batch) in enumerate(dataset_handler):
                # Check the size of the batch
                assert len(input_batch) == len(target_batch)
                if drop_last or idx_batch < len(dataset_handler) - 1:
                    assert len(input_batch) == batch_size
                else:
                    assert len(input_batch) in {num_sequences % batch_size, batch_size}

                # Save the sequences
                for seq in input_batch:
                    input_sequences.append(seq)
                for seq in target_batch:
                    target_sequences.append(seq)

            # Check we have the same number of sequences as expected
            assert len(input_sequences) == len(expected_input_sequences)
            assert len(target_sequences) == len(expected_target_sequences)

            # Check that sequences are not repeated and that all expected sequences exist
            idx_sequences_in_expected_input = set()
            for seq in input_sequences:
                if seq in expected_input_sequences:
                    idx_sequences_in_expected_input.add(expected_input_sequences.index(seq))

            idx_sequences_in_expected_target = set()
            for seq in target_sequences:
                if seq in expected_target_sequences:
                    idx_sequences_in_expected_target.add(expected_target_sequences.index(seq))

            num_seq_dropped = num_sequences % batch_size
            if drop_last and num_seq_dropped != 0:
                assert len(idx_sequences_in_expected_input) == len(idx_sequences_in_expected_target)
                assert batch_size - num_seq_dropped <= len(idx_sequences_in_expected_input) < num_sequences
            else:
                assert len(idx_sequences_in_expected_input) == len(expected_input_sequences)
                assert len(idx_sequences_in_expected_target) == len(expected_target_sequences)