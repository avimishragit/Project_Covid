from setuptools import setup, find_packages

setup(
    name='covid_forecasting_dashboard',
    version='1.0.0',
    description='COVID-19 Forecasting Dashboard with Streamlit',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'prophet',
        'scikit-learn',
        'missingno',
        'joblib',
    ],
    include_package_data=True,
    python_requires='>=3.8',
)
