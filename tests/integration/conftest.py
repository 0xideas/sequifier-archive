import pytest




@pytest.fixture(scope="session")
def project_path():
    return ("tests/project_folder") 