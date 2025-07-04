from project.logging_config import logger
import pandas as pd
from prophet import Prophet

def train_prophet_model(df: pd.DataFrame, params: dict = None):
    """
    Train a Prophet time series forecasting model on the given dataframe.

    Args:
        df (pd.DataFrame): DataFrame with columns 'ds' (date) and 'y' (target value).
        params (dict, optional): Prophet model parameters. Defaults to None.

    Returns:
        Prophet: Trained Prophet model.

    Raises:
        Exception: If model training fails.
    """
    try:
        model = Prophet(**params) if params else Prophet()
        model.fit(df)
        logger.info("Prophet model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training Prophet model: {e}")
        raise

def make_future_dataframe(model, periods=7):
    """
    Create a future dataframe for forecasting with the given Prophet model.

    Args:
        model (Prophet): Trained Prophet model.
        periods (int): Number of periods (days) to forecast into the future. Must be non-negative.

    Returns:
        pd.DataFrame: DataFrame with future dates for prediction.

    Raises:
        ValueError: If periods is negative.
        Exception: If future dataframe creation fails.
    """
    # Raise ValueError if periods is negative (edge case)
    if periods < 0:
        logger.error("Periods must be non-negative.")
        raise ValueError("Periods must be non-negative.")
    try:
        future = model.make_future_dataframe(periods=periods)
        logger.info(f"Future dataframe created for {periods} periods.")
        return future
    except Exception as e:
        logger.error(f"Error creating future dataframe: {e}")
        raise

def predict(model, future_df):
    """
    Generate predictions using a trained Prophet model and a future dataframe.

    Args:
        model (Prophet): Trained Prophet model.
        future_df (pd.DataFrame): DataFrame with future dates (must have 'ds' column).

    Returns:
        pd.DataFrame: Forecast results with columns like 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.

    Raises:
        Exception: If prediction fails.
    """
    try:
        forecast = model.predict(future_df)
        logger.info("Prediction successful.")
        return forecast
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
