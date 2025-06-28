# src/data_preprocessing/PreProcess.py
import os
import pandas as pd

class StockDataPreprocessor:
    def __init__(self, config=None):
        """
        Initializes the StockDataPreprocessor.
        :param config: Configuration dictionary (currently not used by this class directly,
                       but good practice for consistency if future needs arise).
        """
        self.config = config
        # This class primarily operates on paths provided to its methods.

    def run_preprocessing(self, stock_data_dir, preprocessed_data_dir):
        """
        Loads raw stock data CSVs from stock_data_dir, preprocesses them,
        and saves them to preprocessed_data_dir.

        :param stock_data_dir: Directory containing raw stock CSV files.
        :param preprocessed_data_dir: Directory to save preprocessed CSV files.
        :return: List of successfully preprocessed file paths.
        """
        if not os.path.exists(stock_data_dir):
            print(f"Error: Input directory for preprocessing does not exist: {stock_data_dir}")
            return []

        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir, exist_ok=True)
            print(f"Created directory: {preprocessed_data_dir}")

        processed_files = []
        raw_files = [f for f in os.listdir(stock_data_dir) if f.endswith('.csv')]

        if not raw_files:
            print(f"No CSV files found in {stock_data_dir}. Nothing to preprocess.")
            return []

        print(f"[StockDataPreprocessor] Starting preprocessing for {len(raw_files)} files from {stock_data_dir}.")
        print(f"Output directory: {preprocessed_data_dir}")

        for csv_file in raw_files:
            input_filepath = os.path.join(stock_data_dir, csv_file)
            output_filepath = os.path.join(preprocessed_data_dir, csv_file)

            print(f"  Processing file: {csv_file}...")
            try:
                df = pd.read_csv(input_filepath)

                if df.empty:
                    print(f"  Warning: File {csv_file} is empty. Skipping.")
                    # Optionally save an empty file or log this.
                    # For now, just skip creating an output file for it.
                    continue

                # 1. Date Handling
                if 'Date' not in df.columns:
                    print(f"  Error: 'Date' column not found in {csv_file}. Skipping.")
                    continue
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    print(f"  Error parsing 'Date' column in {csv_file}: {e}. Skipping.")
                    continue

                df.sort_values(by='Date', inplace=True) # Ensure data is chronological
                # df.set_index('Date', inplace=True) # Optional: set index for easier time ops, but reset before saving

                # 2. Define essential OHLCV columns
                ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_essential_cols = [col for col in ohlcv_cols if col not in df.columns]
                if missing_essential_cols:
                    print(f"  Error: Missing essential OHLCV columns {missing_essential_cols} in {csv_file}. Skipping.")
                    continue

                # Convert OHLCV to numeric, coercing errors (turns non-numeric to NaN)
                for col in ohlcv_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 3. Handling Missing Values (NaNs)
                # Forward fill, then backward fill for OHLC columns
                ohlc_cols_to_fill = ['Open', 'High', 'Low', 'Close']
                df[ohlc_cols_to_fill] = df[ohlc_cols_to_fill].ffill().bfill()

                # For Volume, fill NaNs with 0, then ffill/bfill (common for non-trading days if they exist in data)
                # Or, one might fill with mean/median if appropriate, but 0 is safer if it means no trades.
                df['Volume'] = df['Volume'].fillna(0).ffill().bfill()

                # 4. Drop rows if 'Close' is still NaN after filling (critical data point)
                # This might happen if the entire column was NaN or only NaNs at start/end.
                df.dropna(subset=['Close'], inplace=True)

                if df.empty:
                    print(f"  Warning: DataFrame for {csv_file} became empty after NaN handling. Skipping output.")
                    continue

                # 5. Data Type Coercion (if not already correct) - yfinance usually good
                # df['Open'] = df['Open'].astype(float)
                # ... similar for High, Low, Close
                # df['Volume'] = df['Volume'].astype(int) # Or float if fractional volumes possible

                # 6. Reset index if 'Date' was set as index, to save it as a column
                # if isinstance(df.index, pd.DatetimeIndex):
                #     df.reset_index(inplace=True)

                # Save the preprocessed DataFrame
                df.to_csv(output_filepath, index=False)
                print(f"  Successfully preprocessed and saved {csv_file} to {output_filepath}")
                processed_files.append(output_filepath)

            except pd.errors.EmptyDataError:
                print(f"  Warning: File {csv_file} is empty or contains no data. Skipping.")
            except Exception as e:
                print(f"  Error processing file {csv_file}: {e}")

        print(f"[StockDataPreprocessor] Finished preprocessing. Successfully processed {len(processed_files)}/{len(raw_files)} files.")
        return processed_files

if __name__ == '__main__':
    print("Running StockDataPreprocessor example...")

    # Create dummy config and directories for testing
    _PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    dummy_config_for_testing = {
        'PROJECT_ROOT': _PROJECT_ROOT_TEST,
        'data_dir': 'data_test_preprocessor',
        'raw_stock_data_dir': 'raw_stock_data_input', # Input for this test
        'processed_stock_data_dir': 'processed_stock_data_output' # Output for this test
    }

    # Define test input and output paths using the dummy config structure
    test_input_base = os.path.join(dummy_config_for_testing['PROJECT_ROOT'],
                                   dummy_config_for_testing['data_dir'])
    test_raw_dir = os.path.join(test_input_base, dummy_config_for_testing['raw_stock_data_dir'])
    test_processed_dir = os.path.join(test_input_base, dummy_config_for_testing['processed_stock_data_dir'])

    # Ensure test directories are clean
    for dir_path in [test_raw_dir, test_processed_dir]:
        if os.path.exists(dir_path):
            for f_name in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f_name))
        else:
            os.makedirs(dir_path)

    # Create some dummy raw CSV files for testing
    dummy_data1 = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'Open': [10, 11, None, 13],
        'High': [10.5, 11.5, 12.5, 13.5],
        'Low': [9.5, 10.5, 11.5, 12.5],
        'Close': [10.2, None, 12.2, 13.2],
        'Volume': [1000, 1100, 0, 1300]
    }
    pd.DataFrame(dummy_data1).to_csv(os.path.join(test_raw_dir, 'TEST1.csv'), index=False)

    dummy_data2 = { # Data with initial NaNs
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'Open': [None, None, 22],
        'High': [None, 21.5, 22.5],
        'Low': [None, 20.5, 21.5],
        'Close': [None, None, 22.2],
        'Volume': [0, 0, 2200]
    }
    pd.DataFrame(dummy_data2).to_csv(os.path.join(test_raw_dir, 'TEST2.csv'), index=False)

    # Empty file
    open(os.path.join(test_raw_dir, 'EMPTY.csv'), 'w').close()

    # File with no Date column
    no_date_data = {'Open': [1,2], 'Close': [1,2]}
    pd.DataFrame(no_date_data).to_csv(os.path.join(test_raw_dir, 'NODATE.csv'), index=False)


    preprocessor = StockDataPreprocessor(config=dummy_config_for_testing) # Config not strictly used by class methods here

    print(f"Test input raw directory: {test_raw_dir}")
    print(f"Test output processed directory: {test_processed_dir}")

    processed_file_list = preprocessor.run_preprocessing(test_raw_dir, test_processed_dir)
    print(f"Files successfully processed in test: {processed_file_list}")

    # Verify processed files (basic check)
    for file_path in processed_file_list:
        if os.path.exists(file_path):
            print(f"Verified: File '{file_path}' exists.")
            df_check = pd.read_csv(file_path)
            print(f"Data for {os.path.basename(file_path)} (first 3 rows):
{df_check.head(3)}")
            # Check for NaNs in critical columns
            if df_check[['Open', 'High', 'Low', 'Close']].isnull().values.any():
                 print(f"  Warning: Found NaNs in OHLC of {os.path.basename(file_path)} after preprocessing.")
            if df_check['Volume'].isnull().values.any():
                 print(f"  Warning: Found NaNs in Volume of {os.path.basename(file_path)} after preprocessing.")
        else:
            print(f"Error: Processed file '{file_path}' NOT found.")

    print("StockDataPreprocessor example finished.")
