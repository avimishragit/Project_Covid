import plotly.express as px
import plotly.graph_objects as go

def plot_forecast(model, forecast, title="Forecast"):
    from prophet.plot import plot_plotly
    fig = plot_plotly(model, forecast)
    fig.update_layout(title=title, width=1200, height=800)
    fig.show()

def plot_components(model, forecast):
    from prophet.plot import plot_components_plotly
    fig = plot_components_plotly(model, forecast)
    fig.show()
