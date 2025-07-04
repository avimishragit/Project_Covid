import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data_original')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'covid.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'Models')
PARAMS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Model_parameters')
FORECAST_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data_modified')

# Prophet hyperparameter grid
PROPHET_PARAM_GRID = {
    "changepoint_prior_scale": [0.001, 0.12575, 0.2505, 0.37525, 0.5],
    "seasonality_prior_scale": [0.01, 0.7575, 1.505, 2.2525, 3.0],
    "seasonality_mode": ['additive', 'multiplicative'],
    "changepoint_range": [0.5, 0.6125, 0.725, 0.8375, 0.95],
}
