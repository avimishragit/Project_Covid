import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import joblib
import json
from logging_config import logger
import dotenv

from pathlib import Path

# --- Path Setup for Robustness ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'Data_modified')
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

PARAMS_DIR = os.path.join(BASE_DIR, 'Model_parameters')

# --- About Section Content ---
def get_readme_content():
    readme_path = Path(BASE_DIR) / "README.md"
    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "README.md not found."

# Load environment variables from .env file
dotenv.load_dotenv(os.path.join(BASE_DIR, '.env'))

def about_section():
    st.title("About This Project")
    st.markdown(get_readme_content())

# --- Helper Functions ---
def get_available_countries():
    try:
        files = os.listdir(DATA_DIR)
        countries = set()
        for f in files:
            if '_' in f:
                country = f.split('_')[0]
                countries.add(country)
        logger.info(f"Loaded countries: {countries}")
        return sorted(list(countries))
    except Exception as e:
        logger.error(f"Error loading countries: {e}")
        st.error("Failed to load available countries. Please check your data directory.")
        return []

def get_available_types(country):
    try:
        files = os.listdir(DATA_DIR)
        types = set()
        for f in files:
            if f.startswith(country + '_') and f.endswith('_weekly_forecast.csv'):
                t = f.split('_')[1]
                types.add(t)
        logger.info(f"Loaded types for {country}: {types}")
        return sorted(list(types))
    except Exception as e:
        logger.error(f"Error loading types for {country}: {e}")
        st.error(f"Failed to load available types for {country}.")
        return []

def load_forecast(country, case_type):
    path = os.path.join(DATA_DIR, f'{country}_{case_type}_weekly_forecast.csv')
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"Loaded forecast for {country} - {case_type}")
            return df
        else:
            logger.warning(f"Forecast file not found: {path}")
            return None
    except Exception as e:
        logger.error(f"Error loading forecast for {country} - {case_type}: {e}")
        st.error(f"Failed to load forecast for {country} - {case_type}.")
        return None

# --- Main App Navigation ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "About"])
    if page == "About":
        about_section()
        return
    # ...existing dashboard code...

# --- Run App ---
if __name__ == "__main__":
    main()

def plot_forecast_graph(df, country, case_type, key=None):
    try:
        fig = px.line(df, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'],
                      labels={'ds': 'Date', 'value': 'Predicted'},
                      title=f"7-Day Forecast for {country} - {case_type}")
        st.plotly_chart(fig, use_container_width=True, key=key)
        logger.info(f"Plotted forecast graph for {country} - {case_type}")
    except Exception as e:
        logger.error(f"Error plotting forecast graph for {country} - {case_type}: {e}")
        st.error(f"Failed to plot forecast for {country} - {case_type}.")

# Helper to get available models
@st.cache_data
def get_model_files():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_prophet_model.joblib')]
    return files

@st.cache_data
def get_param_files():
    files = [f for f in os.listdir(PARAMS_DIR) if f.endswith('_model_params.json')]
    return files

@st.cache_data
def get_forecast_files():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('_weekly_forecast.csv')]
    return files

# Load model and params
def load_model_and_params(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    param_name = model_name.replace('_prophet_model.joblib', '_model_params.json')
    param_path = os.path.join(PARAMS_DIR, param_name)
    model = joblib.load(model_path)
    params = None
    if os.path.exists(param_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
    return model, params

# Load forecast
def load_forecast_csv(forecast_name):
    path = os.path.join(DATA_DIR, forecast_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- Streamlit App ---
st.set_page_config(page_title="COVID-19 Forecasting Dashboard", layout="wide")

# --- Sidebar Navigation ---
pages = [
    "Project Overview & Chatbot",
    "Country Prediction",
    "Project Graphs"
]
page = st.sidebar.radio("Navigation", pages)

# --- 1. Project Overview & Chatbot ---
if page == "Project Overview & Chatbot":
    st.title("COVID-19 Forecasting Project")
    st.markdown("""
    This dashboard provides:
    - Interactive COVID-19 forecasts for each country and case type
    - Visualizations and trends from the project
    - An AI-powered chatbot for project Q&A
    """)
    st.header("Ask the Project Chatbot")
    from project.chatbot_gemini import ask_gemini
    user_q = st.text_input("Ask a question about the project, data, or COVID-19 trends:")
    if user_q:
        with st.spinner("Gemini is thinking..."):
            try:
                answer = ask_gemini(user_q)
                st.success(answer)
            except Exception as e:
                st.error(f"Chatbot error: {e}")
    else:
        st.info("This is a Gemini LLM-powered chatbot with access to project data and the web.")

# --- 2. Country Prediction ---
elif page == "Country Prediction":
    st.title("COVID-19 Prediction by Country and Case Type")
    countries = get_available_countries()
    country = st.selectbox("Select Country/Region", countries)
    types = get_available_types(country)
    case_type = st.selectbox("Select Case Type", types)
    df = load_forecast(country, case_type)
    if df is not None:
        st.subheader(f"Forecast Table for {country} - {case_type}")
        st.dataframe(df)
        plot_forecast_graph(df, country, case_type)
    else:
        st.warning("No forecast data available for this selection.")

# --- 3. Project Graphs ---
elif page == "Project Graphs":
    st.title("Project Visualizations & Trends")
    st.header("Global Trends")
    global_df = load_forecast('global', 'Confirmed')
    if global_df is not None:
        plot_forecast_graph(global_df, 'global', 'Confirmed', key='global-Confirmed')
    st.header("Country/Region Trends")
    for country in get_available_countries():
        for case_type in get_available_types(country):
            st.subheader(f"{country} - {case_type}")
            df = load_forecast(country, case_type)
            if df is not None:
                plot_forecast_graph(df, country, case_type, key=f"{country}-{case_type}")

