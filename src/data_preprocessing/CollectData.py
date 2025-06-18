import os
import pandas as pd
class StockDataCollector:
    def __init__(self, save_dir): self.save_dir = save_dir
    def run_collect(self, ticker_list, start_date, end_date):
        print(f"[Dummy CollectData] Called with tickers: {ticker_list}, save_dir: {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        for ticker in ticker_list:
            dummy_file = os.path.join(self.save_dir, f"{ticker}.csv")
            pd.DataFrame({'Date': pd.to_datetime(['2020-01-01']), 'Close': [100]}).to_csv(dummy_file, index=False)
            print(f"[Dummy CollectData] Created dummy file: {dummy_file}")
