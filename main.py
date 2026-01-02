import pandas as pd
import numpy as np
import yfinance as yf

def load_stock_data(ticker, start_date, end_date):
    
    ticker = yf.Ticker(ticker)
    df = ticker.history(start=start_date, end=end_date)

    if not df.empty:
        print(f"loaded data from {ticker} from {start_date} to {end_date}")
        return df
    else: 
        print("could not return data")
        return None
