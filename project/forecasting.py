import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics

def cross_validate_prophet(model, initial='120 days', period='30 days', horizon='10 days'):
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel='processes')
    df_p = performance_metrics(df_cv, rolling_window=3)
    return df_cv, df_p
