import yfinance as yf
import pandas as pd
import os
import numpy as np

class StockDataCollector:
    
    def __init__(self, save_dir='stock_data'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")

    def run_collect(self, tickers, start_date, end_date):
        for ticker in tickers:
            try:
                self._process_ticker(ticker, start_date, end_date)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    def _process_ticker(self, ticker, start_date, end_date):
        print(f"Processing {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data available for {ticker}")
            return
        
        # Fetch financial metrics
        ticker_info = yf.Ticker(ticker).info
        eps = ticker_info.get('trailingEps')
        pe_ratio = ticker_info.get('trailingPE')
        
        # Add financial metrics as columns
        data['EPS'] = eps
        data['PE_Ratio'] = pe_ratio
        
        # Calculate 30-day historical volatility (annualized)
        close_prices = data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1))
        data['Volatility_30D'] = log_returns.rolling(30).std() * np.sqrt(252)
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'EPS', 
                            'PE_Ratio', 'Volatility_30D']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            print(f"Error: Missing columns for {ticker}: {missing}")
            return
        
        # Save to CSV
        filename = os.path.join(self.save_dir, f"{ticker}.csv")
        data.to_csv(filename, index=True)
        print(f"Data saved for {ticker} to {filename}")

