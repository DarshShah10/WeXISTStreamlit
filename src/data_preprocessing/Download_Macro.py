# src/data_preprocessing/Download_Macro.py
import os
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta

class MacroDataDownloader:
    def __init__(self, config=None):
        """
        Initializes the MacroDataDownloader.
        :param config: Configuration dictionary.
        """
        self.config = config
        if not self.config:
            # This component heavily relies on config for feature list and dates.
            # A default or dummy config could be loaded here if essential for standalone testing,
            # but in pipeline context, config should always be provided.
            print("Warning: MacroDataDownloader initialized without a config. May use hardcoded defaults for testing.")
            # Example minimal config for standalone testing:
            self.config = {
                'default_start_date': (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
                'default_end_date': datetime.now().strftime('%Y-%m-%d'),
                'macro_feature_list': ['snp500', 'gold', 'dgs10', 'vix'], # Matches keys in _get_macro_data
                'PROJECT_ROOT': '.', # Assume current dir is project root for standalone test
                'data_dir': 'data_test_macro',
                'macro_data_filename': 'macro_economic_data_test.csv'
            }


    def _get_macro_data(self, feature_name, start_date, end_date):
        """
        Fetches data for a single macroeconomic feature.
        """
        print(f"  Fetching macro feature: {feature_name} from {start_date} to {end_date}...")
        data = None
        try:
            if feature_name == 'snp500':
                data = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=True)[['Close']]
                data.rename(columns={'Close': 'snp500'}, inplace=True)
            elif feature_name == 'gold':
                data = yf.download('GC=F', start=start_date, end=end_date, progress=False, auto_adjust=True)[['Close']]
                data.rename(columns={'Close': 'gold'}, inplace=True) # Config uses 'gold_price', adjust if needed
            elif feature_name == 'dgs10': # 10-Year Treasury Constant Maturity Rate
                data = web.DataReader('DGS10', 'fred', start_date, end_date)
                data.rename(columns={'DGS10': 'dgs10'}, inplace=True)
            elif feature_name == 'vix': # Volatility Index
                data = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)[['Close']]
                data.rename(columns={'Close': 'vix'}, inplace=True)
            # Add more features here as needed, ensure they match config['macro_feature_list']
            else:
                print(f"  Macro feature '{feature_name}' not recognized or download logic not implemented.")
                return None

            if data is not None and not data.empty:
                print(f"  Successfully fetched {feature_name}.")
                return data
            else:
                print(f"  No data found for {feature_name}.")
                return None
        except Exception as e:
            print(f"  Error fetching {feature_name}: {e}")
            return None

    def run_download_macro(self, macro_data_file_path, start_date_str=None, end_date_str=None):
        """
        Downloads various macroeconomic indicators and saves them to a single CSV file.

        :param macro_data_file_path: Full path to save the combined macro data CSV.
        :param start_date_str: Optional. Start date (YYYY-MM-DD). Defaults to config's default_start_date.
        :param end_date_str: Optional. End date (YYYY-MM-DD). Defaults to config's default_end_date.
        :return: Path to the saved CSV file if successful, else None.
        """
        output_dir = os.path.dirname(macro_data_file_path)
        if output_dir and not os.path.exists(output_dir): # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        # Determine date range
        start_date = start_date_str if start_date_str else self.config.get('default_start_date')
        end_date_val = end_date_str if end_date_str else self.config.get('default_end_date')

        # Adjust end_date for yfinance/pandas_datareader (similar to StockDataCollector)
        try:
            end_date_dt = datetime.strptime(end_date_val, '%Y-%m-%d')
            adjusted_end_date = (end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid end_date format '{end_date_val}'. Using original end_date.")
            adjusted_end_date = end_date_val

        if not start_date or not end_date_val:
            print("Error: Start or end date for macro data download not provided or found in config.")
            return None

        print(f"[MacroDataDownloader] Starting download for macro features.")
        print(f"Output file: {macro_data_file_path}")
        print(f"Date range: {start_date} to {end_date_val} (adjusted end for download: {adjusted_end_date})")

        feature_list = self.config.get('macro_feature_list', ['snp500', 'gold', 'dgs10', 'vix']) # Default if not in config
        if not feature_list:
            print("No macro features specified in config. Nothing to download.")
            return None

        all_macro_data = []

        for feature in feature_list:
            data = self._get_macro_data(feature, start_date, adjusted_end_date)
            if data is not None:
                all_macro_data.append(data)

        if not all_macro_data:
            print("No macro data could be downloaded for any specified feature.")
            # Create an empty file with Date column? Or return None.
            # pd.DataFrame(columns=['Date']).to_csv(macro_data_file_path, index=False)
            return None

        # Combine all DataFrames. They should all have a DatetimeIndex.
        combined_df = pd.concat(all_macro_data, axis=1)

        # Ensure 'Date' is a column if it's in the index (it should be from yf/web)
        if isinstance(combined_df.index, pd.DatetimeIndex) and 'Date' not in combined_df.columns:
            combined_df.reset_index(inplace=True)
            combined_df.rename(columns={'index': 'Date'}, inplace=True) # if index had no name
            combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.normalize()


        # Fill missing values: forward fill first, then backward fill
        # This is common for macro data as different series might have missing values on different days (e.g. holidays)
        # DGS10, for example, is not updated daily.
        combined_df.sort_values(by='Date', inplace=True) # Ensure sorted by date before ffill/bfill
        # Set Date as index temporarily for ffill/bfill to work correctly across series, then reset
        if 'Date' in combined_df.columns:
            combined_df.set_index('Date', inplace=True)

        combined_df = combined_df.ffill().bfill()

        if isinstance(combined_df.index, pd.DatetimeIndex) and 'Date' not in combined_df.columns:
            combined_df.reset_index(inplace=True) # Ensure Date is a column for saving

        try:
            combined_df.to_csv(macro_data_file_path, index=False)
            print(f"Successfully saved combined macro data to {macro_data_file_path}")
            return macro_data_file_path
        except Exception as e:
            print(f"Error saving combined macro data to {macro_data_file_path}: {e}")
            return None

if __name__ == '__main__':
    print("Running MacroDataDownloader example...")

    # Create a dummy config for testing
    # Note: PROJECT_ROOT should point to the actual project root for config file loading if this was a real run.
    # For this __main__ test, paths are relative to this file or use explicit test dirs.
    _PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    dummy_config_for_testing = {
        'PROJECT_ROOT': _PROJECT_ROOT_TEST,
        'data_dir': 'data_test_macro_downloader', # Test-specific data directory
        'macro_data_filename': 'test_macro_data.csv',
        'default_start_date': (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'), # 2 years of data
        'default_end_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        'macro_feature_list': ['snp500', 'gold', 'dgs10', 'vix', 'unimplemented_feature']
    }

    downloader = MacroDataDownloader(config=dummy_config_for_testing)

    # Construct test output path
    test_data_dir = os.path.join(dummy_config_for_testing['PROJECT_ROOT'], dummy_config_for_testing['data_dir'])
    test_output_file = os.path.join(test_data_dir, dummy_config_for_testing['macro_data_filename'])

    # Clean up old test file if it exists
    if os.path.exists(test_output_file):
        os.remove(test_output_file)
    if not os.path.exists(test_data_dir):
         os.makedirs(test_data_dir)

    print(f"Test output file will be: {test_output_file}")

    # Use specific dates for the test run, overriding config defaults for this call
    test_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d') # Shorter period for faster test
    test_end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    saved_path = downloader.run_download_macro(test_output_file, start_date_str=test_start, end_date_str=test_end)

    if saved_path and os.path.exists(saved_path):
        print(f"Macro data download test successful. File saved to: {saved_path}")
        df_check = pd.read_csv(saved_path)
        print(f"Downloaded data shape: {df_check.shape}")
        print(f"Columns: {df_check.columns.tolist()}")
        print(f"Data (first 3 rows):
{df_check.head(3)}")
        print(f"Data (last 3 rows):
{df_check.tail(3)}")
    else:
        print("Macro data download test failed or file not saved.")

    print("MacroDataDownloader example finished.")
