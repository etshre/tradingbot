import pandas as pd
import numpy as np
import yfinance as yf
from env import StockTradingEnv
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


def load_stock_data(ticker_symbol, start_date, end_date):
    
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date)

    if not df.empty:
        print(f"loaded data from {ticker_symbol} from {start_date} to {end_date}")
        return df
    else: 
        print("could not return data")
        return None
    
    df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')


TICKER = 'NVDA'
START_DATE = '2000-01-01'
END_DATE = '2026-01-01'

stock_df = load_stock_data(TICKER, START_DATE, END_DATE)

if stock_df is None:
    print(stock_df.head())

def preprocess_data(df):
    df['Close_Norm'] = df['Close'] / df['Close'].max()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].max()

    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df.fillna(method='bfill', inplace = True)

    features_df = df[['Close_Norm', 'Volume_Norm', 'SMA50']]
    return features_df

if stock_df is not None:
    processed_df = preprocess_data(stock_df)
    print('\nProccessed Data with new features:')
    print(processed_df.head())