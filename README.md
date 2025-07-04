
# COVID-19 Forecasting, Visualization & AI Chatbot Dashboard

This project is a production-ready, modular pipeline for COVID-19 data analysis, time series forecasting, interactive visualization, and AI-powered Q&A. It leverages Python, Prophet, Streamlit, and a Gemini+LangChain chatbot with web search fallback for a complete data science and analytics experience.


## Key Features
- **Data Cleaning & Preprocessing:** Modular scripts for robust cleaning, grouping, and feature engineering of COVID-19 data by country/region.
- **Time Series Modeling:** Prophet-based models for confirmed, deaths, recovered, and active cases, with parameter saving and evaluation.
- **Forecasting:** Automated generation and storage of future predictions for each country, region, and globally.
- **Interactive Visualization:** Streamlit dashboard with:
  - Project overview and About section
  - Country/type selection and forecast graphs
  - Model/parameter inspection
  - All project trends and visualizations
- **AI Chatbot (Gemini+LangChain):**
  - Ask questions about the project, data, COVID-19 trends, or modeling
  - If the LLM doesn't know, it automatically performs a DuckDuckGo web search and returns the result
  - Context-aware: uses project details and a sample of the original data for more accurate answers
- **Logging:** All scripts log actions and errors to `logs/app.log` for easy debugging and monitoring.
- **Testing:** Comprehensive unit tests for modeling functions, including edge cases.


## Project Structure
```
Covid/
├── Data_modified/           # Processed and forecasted data CSVs (per country/region/type)
├── Data_original/           # Raw COVID-19 data (e.g., covid.csv)
├── logs/                    # Log files (app.log, etc.)
├── Model_parameters/        # Saved model parameter JSONs (per model/country/type)
├── Models/                  # Trained Prophet models (joblib files)
├── notebooks/               # Jupyter notebooks for exploration and prototyping
├── project/                 # All main Python scripts and modules
│   ├── config.py                # Global config and constants
│   ├── data_acquisition.py      # Data download and ingestion
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── eda.py                   # Exploratory data analysis
│   ├── modeling.py              # Prophet modeling functions
│   ├── evaluation.py            # Model evaluation metrics
│   ├── forecasting.py           # Forecasting logic
│   ├── visualization.py         # Plotting and visualization utilities
│   ├── pipeline.py              # End-to-end pipeline orchestration
│   ├── streamlit_app.py         # Streamlit dashboard app
│   ├── chatbot_gemini.py        # Gemini+LangChain chatbot integration
│   ├── web_search_agent.py      # DuckDuckGo web search fallback for chatbot
│   ├── logging_config.py        # Logging setup
│   ├── test_modeling.py         # Unit tests for modeling
│   └── setup.py                 # Packaging/setup script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not committed)
├── README.md                # Project documentation
└── ...
```

## Quickstart
1. **Clone the repository and navigate to the folder:**
   ```sh
   cd path/to/Covid
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # Or: source .venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up your environment variables:**
   - Create a `.env` file in the project root with your Gemini API key:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
   - (Optional) Add other API keys as needed for web search, etc.
5. **Run tests:**
   ```sh
   python -m unittest project/test_modeling.py
   ```
6. **Run the Streamlit dashboard:**
   ```sh
   streamlit run project/streamlit_app.py
   ```


## Usage
- Launch the dashboard to explore forecasts, trends, and model details interactively.
- Use the "About" section for a full project overview.
- Ask the AI chatbot anything about the project, COVID-19 data, or time series forecasting—if the LLM doesn't know, it will search the web for you!
- All logs are saved in `logs/app.log`.
- Modify or extend scripts in `project/` for custom analysis or new features.


## Extending the Project
- Add new models, features, or dashboard pages by creating new scripts in `project/`.
- Add more tests in the `project/` folder for robust development.
- Tune the chatbot prompt or swap in other LLMs for advanced Q&A.


## Authors
- Avinash Mishra


## License
This project is licensed under the MIT License.
