def add_lag_features(df, columns, lag=1):
    for col in columns:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def add_date_features(df):
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df
