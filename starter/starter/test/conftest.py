import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data(request):
    data = pd.read_csv("starter/data/census.csv")
    return data
