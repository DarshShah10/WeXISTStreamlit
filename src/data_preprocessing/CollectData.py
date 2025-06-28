# src/data_preprocessing/CollectData.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class StockDataCollector:
    def __init__(self, save_dir=None, config=None):
        """
        Initializes the StockDataCollector.
        :param save_dir: Directory to save downloaded CSV files. If None, derived from config.
        :param config: Configuration dictionary.
        """
        self.config = config
        self.save_dir = save_dir

        if self.save_dir is None and self.config:
            project_root = self.config.get('PROJECT_ROOT', '.')
            data_dir = self.config.get('data_dir', 'data')
            raw_stock_subdir = self.config.get('raw_stock_data_dir', 'raw_stock_data')
            self.save_dir = os.path.join(project_root, data_dir, raw_stock_subdir)
        elif self.save_dir is None:
            # Fallback if no config and no save_dir provided (should not happen in normal flow)
            self.save_dir = os.path.join(os.getcwd(), "data", "raw_stock_data")
            print(f"Warning: StockDataCollector save_dir fell back to default: {self.save_dir}")

        if not self.save_dir: # Should be caught by above, but as a safeguard
            raise ValueError("Save directory for StockDataCollector could not be determined.")

        # Ensure save_dir is absolute path
        self.save_dir = os.path.abspath(self.save_dir)


    def run_collect(self, ticker_list, start_date_str, end_date_str):
        """
        Downloads historical stock data for the given tickers and date range.
        Saves data for each ticker to a CSV file in self.save_dir.

        :param ticker_list: List of stock tickers (e.g., ["AAPL", "MSFT"]).
        :param start_date_str: Start date for data collection (YYYY-MM-DD string).
        :param end_date_str: End date for data collection (YYYY-MM-DD string).
        :return: List of tickers for which data was successfully downloaded and saved.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Created directory: {self.save_dir}")

        successful_tickers = []
        if not ticker_list:
            print("StockDataCollector: No tickers provided. Nothing to collect.")
            return successful_tickers

        # yfinance typically downloads data up to, but not including, the end_date.
        # To include the end_date, add one day to it for the yf.download call.
        try:
            end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            adjusted_end_date_str = (end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid end_date format '{end_date_str}'. Please use YYYY-MM-DD. Using original end_date.")
            adjusted_end_date_str = end_date_str


        print(f"[StockDataCollector] Starting data collection for {len(ticker_list)} tickers.")
        print(f"Output directory: {self.save_dir}")
        print(f"Date range: {start_date_str} to {end_date_str} (adjusted end for download: {adjusted_end_date_str})")

        for ticker in ticker_list:
            output_filepath = os.path.join(self.save_dir, f"{ticker.replace('.', '_')}.csv") # Sanitize ticker for filename
            print(f"  Fetching data for {ticker}...")
            try:
                # Download data; yfinance progress bar can be noisy, suppress with progress=False
                # yf.download can take a list of tickers, but saving individually gives more control.
                data_df = yf.download(ticker, start=start_date_str, end=adjusted_end_date_str, progress=False, auto_adjust=True)

                if data_df.empty:
                    print(f"  No data found for {ticker} for the given period.")
                    # Create an empty file to signify an attempt was made, or skip.
                    # For now, skip creating empty file to avoid issues downstream if not handled.
                    # with open(output_filepath, 'w') as f: # Create empty file
                    #    f.write("Date,Open,High,Low,Close,Volume
") # Example header for empty CSV
                    # print(f"  Created empty file for {ticker} at {output_filepath} as no data was returned.")

                else:
                    # Ensure 'Date' is a column if it's in the index
                    if isinstance(data_df.index, pd.DatetimeIndex) and 'Date' not in data_df.columns:
                        data_df.reset_index(inplace=True) # Moves 'Date' from index to a column

                    # Standardize column names to ensure consistency (yfinance usually returns them capitalized)
                    data_df.rename(columns={
                        'Date': 'Date', 'Open': 'Open', 'High': 'High',
                        'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                        # Adjusted Close is usually handled by auto_adjust=True
                    }, inplace=True)

                    # Select only standard OHLCV columns if others exist (e.g. Dividends, Stock Splits if auto_adjust=False)
                    # With auto_adjust=True, these are usually not present.
                    # standard_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    # data_df = data_df[[col for col in standard_cols if col in data_df.columns]]


                    data_df.to_csv(output_filepath, index=False)
                    print(f"  Successfully saved data for {ticker} to {output_filepath}")
                    successful_tickers.append(ticker)

            except Exception as e:
                print(f"  Error collecting data for {ticker}: {e}")
                # Optionally, create an empty file or a file with error info
                # with open(output_filepath.replace('.csv', '_error.txt'), 'w') as f:
                #     f.write(str(e))

        print(f"[StockDataCollector] Finished data collection. Successfully processed {len(successful_tickers)}/{len(ticker_list)} tickers.")
        return successful_tickers

if __name__ == '__main__':
    # Example Usage (requires shared_config.yaml to be present in ../../config relative to this file for default config loading)
    print("Running StockDataCollector example...")

    # Create a dummy config for testing if shared_config.yaml is not set up for this specific test
    dummy_config_for_testing = {
        'PROJECT_ROOT': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), # project root
        'data_dir': 'data_test_collector', # Use a test-specific data directory
        'raw_stock_data_dir': 'raw_stock_data_test'
    }

    # Ensure the test directory exists and is clean for the test
    test_output_dir = os.path.join(dummy_config_for_testing['PROJECT_ROOT'],
                                   dummy_config_for_testing['data_dir'],
                                   dummy_config_for_testing['raw_stock_data_dir'])
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    else: # Clean up old files from previous test run
        for f_name in os.listdir(test_output_dir):
            os.remove(os.path.join(test_output_dir, f_name))


    collector = StockDataCollector(config=dummy_config_for_testing)

    # Test with a few tickers
    test_tickers = ['AAPL', 'GOOGL', 'NONEXISTENTTICKER']
    # Using a more recent but short period for quick test
    test_start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    test_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Test Output Directory: {collector.save_dir}")

    results = collector.run_collect(test_tickers, test_start_date, test_end_date)
    print(f"Tickers successfully processed in test: {results}")

    # Verify files were created (basic check)
    for ticker in results:
        expected_file = os.path.join(collector.save_dir, f"{ticker.replace('.', '_')}.csv")
        if os.path.exists(expected_file):
            print(f"Verified: File '{expected_file}' exists.")
            # df_check = pd.read_csv(expected_file)
            # print(f"Data for {ticker} (first 5 rows):
{df_check.head()}")
        else:
            print(f"Error: File '{expected_file}' NOT found.")

    print("StockDataCollector example finished.")
