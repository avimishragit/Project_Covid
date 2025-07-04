from .data_acquisition import load_covid_data
from .data_preprocessing import clean_data, preprocess_grouped_data
from .feature_engineering import add_lag_features, add_date_features
from .modeling import train_prophet_model, make_future_dataframe, predict
from .evaluation import evaluate_forecast
from .config import PROPHET_PARAM_GRID
from .utils import save_model, save_params
import os

def run_pipeline():
    # 1. Load and clean data
    df = load_covid_data()
    df = clean_data(df)

    # 2. Preprocess for global, region, and country
    global_data = preprocess_grouped_data(df)['global']
    region_data = preprocess_grouped_data(df, 'WHO Region')
    country_data = preprocess_grouped_data(df, 'Country/Region')

    # 3. Feature engineering
    for d in [global_data] + list(region_data.values()) + list(country_data.values()):
        d = add_lag_features(d, ['Confirmed', 'Deaths', 'Recovered', 'Active'])
        d = add_date_features(d)

    # 4. Modeling and saving for each group/target
    results = {}
    for name, data in {'global': global_data, **region_data, **country_data}.items():
        for target in ['Confirmed', 'Deaths', 'Recovered', 'Active']:
            df_model = data.reset_index()[['Date', target]].rename(columns={'Date': 'ds', target: 'y'})
            model = train_prophet_model(df_model)
            future = make_future_dataframe(model, periods=7)
            forecast = predict(model, future)
            # Save model and forecast
            model_path = os.path.join('..', 'Models', f'{name}_{target}_prophet_model.joblib')
            save_model(model, model_path)
            # Optionally save forecast as CSV
            forecast_path = os.path.join('..', 'Data_modified', f'{name}_{target}_weekly_forecast.csv')
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_csv(forecast_path, index=False)
            results[f'{name}_{target}'] = forecast
    # Optionally return results for further use
    return results

if __name__ == "__main__":
    run_pipeline()
