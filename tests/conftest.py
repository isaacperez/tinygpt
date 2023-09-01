from pathlib import Path
import pytest


@pytest.fixture()
def TextDataset_files() -> tuple[Path, Path]:
    current_folder = Path(__file__).parent

    metadata_file_path = current_folder / 'test_metadata.json'
    data_file_path = current_folder / 'test_data.txt'

    return data_file_path, metadata_file_path
