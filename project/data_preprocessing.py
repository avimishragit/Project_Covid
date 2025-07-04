import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the COVID-19 data.
    Steps performed:
        - Drop 'Province/State' column if present
        - Convert 'Date' column to datetime
        - Downcast int64 columns to int32 for memory efficiency
        - Convert object columns to category dtype
        - Remove duplicate rows
    Args:
        df (pd.DataFrame): Raw COVID-19 data
    Returns:
        pd.DataFrame: Cleaned and preprocessed data
    """
    df = df.copy()
    if 'Province/State' in df.columns:
        df = df.drop(['Province/State'], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    for col in df.select_dtypes(include='int64').columns:
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    df = df.drop_duplicates()
    return df

def preprocess_grouped_data(df: pd.DataFrame, group_col: str = None) -> dict:
    """
    Group data by a column (e.g., region or country) and preprocess each group.
    For each group:
        - Convert 'Date' to datetime
        - Set 'Date' as index and resample daily, summing values
        - Drop 'Lat' and 'Long' columns if present
    Args:
        df (pd.DataFrame): Cleaned COVID-19 data
        group_col (str, optional): Column to group by (e.g., 'Country/Region'). If None, process as global.
    Returns:
        dict: Dictionary of preprocessed DataFrames, keyed by group name (or 'global')
    """
    datasets = {}
    if group_col:
        for key, group in df.groupby(group_col):
            data = group.copy()
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date').resample('D').sum()
            if 'Lat' in data.columns and 'Long' in data.columns:
                data = data.drop(['Lat', 'Long'], axis=1)
            datasets[key] = data
    else:
        data = df.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date').resample('D').sum()
        if 'Lat' in data.columns and 'Long' in data.columns:
            data = data.drop(['Lat', 'Long'], axis=1)
        datasets['global'] = data
    return datasets
