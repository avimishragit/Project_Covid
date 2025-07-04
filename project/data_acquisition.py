import pandas as pd
from .config import RAW_DATA_PATH

def load_covid_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load COVID-19 data from CSV."""
    df = pd.read_csv(path)
    return df
