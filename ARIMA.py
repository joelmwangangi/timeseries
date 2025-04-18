import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import plotly.graph_objects as go

# Function to fetch data, fit ARIMA model, and display results
def run_arima_model():
    try:
        # Get user inputs from Streamlit form
        ticker_symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL):", "AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))
        p = st.number_input("ARIMA(p) Parameter:", min_value=0, value=5)
        d = st.number_input("ARIMA(d) Parameter:", min_value=0, value=1)
        q = st.number_input("ARIMA(q) Parameter:", min_value=0, value=5)
        
        # Download stock data
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        # Fit the ARIMA model
        model = ARIMA(stock_data['Close'], order=(p, d, q))
        model_fit = model.fit()
        
        # Show the model summary
        st.subheader("ARIMA Model Summary")
        st.text(model_fit.summary())
        
        # Forecast the next 30 days
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)
        future_dates = pd.date_range(start=stock_data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
        
        # Plotting the data and forecast with Plotly
        fig = go.Figure()

        # Add actual stock prices
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Stock Prices'))

        # Add forecasted values
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecasted Stock Prices', line=dict(color='red')))

        # Set plot titles and labels
        fig.update_layout(
            title=f'{ticker_symbol} Stock Price Forecast (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Closing Price (USD)',
            template='plotly_dark'
        )

        # Show the interactive plot
        st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app layout
st.title("ARIMA Model for Stock Price Forecasting")
st.write("""
    This app allows you to input a stock ticker, select a date range, 
    and apply an ARIMA model to forecast future stock prices.
""")

# Run the model
run_arima_model()
