from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_forecast(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "mape": mape, "accuracy_range": (100 - mape, 100 + mape)}
