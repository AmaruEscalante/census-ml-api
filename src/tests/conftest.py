import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data(request):
    data = pd.read_csv("src/data/census.csv")
    return data
