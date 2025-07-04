import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def plot_missing_data(df: pd.DataFrame):
    import missingno as msno
    msno.matrix(df)
    plt.show()

def plot_country_distribution(df: pd.DataFrame):
    fig = go.Figure(data=[go.Pie(labels=df['Country/Region'].value_counts().index,
                                 values=df['Country/Region'].value_counts().values, hole=0.4)])
    fig.update_layout(title="Countries Distribution")
    fig.show()

def plot_time_series(df: pd.DataFrame, columns, title="Time Series"):
    fig = px.line(df, x=df.index, y=columns, title=title)
    fig.show()
