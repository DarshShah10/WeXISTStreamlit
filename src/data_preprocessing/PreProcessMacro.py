# src/data_preprocessing/PreProcessMacro.py
import os
import pandas as pd

class MacroDataProcessor:
    def __init__(self, config=None):
        """
        Initializes the MacroDataProcessor.
        :param config: Configuration dictionary. Used for logging or future enhancements.
        """
        self.config = config
        # This class primarily operates on paths provided to its methods.

    def run_macro_postProcess(self, raw_macro_data_path, processed_macro_data_path):
        """
        Loads the raw macro data CSV, performs preprocessing, and saves it.
        For now, preprocessing is minimal as MacroDataDownloader already does ffill/bfill.
        This step ensures date formatting and column consistency.

        :param raw_macro_data_path: Path to the input CSV file from MacroDataDownloader.
        :param processed_macro_data_path: Path to save the processed macro data CSV.
        :return: Path to the saved processed CSV file if successful, else None.
        """
        if not os.path.exists(raw_macro_data_path):
            print(f"Error: Raw macro data file not found: {raw_macro_data_path}")
            return None

        output_dir = os.path.dirname(processed_macro_data_path)
        if output_dir and not os.path.exists(output_dir): # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        print(f"[MacroDataProcessor] Starting preprocessing for macro data file: {raw_macro_data_path}")
        print(f"Output file: {processed_macro_data_path}")

        try:
            df_macro = pd.read_csv(raw_macro_data_path)

            if df_macro.empty:
                print(f"  Warning: Raw macro data file {raw_macro_data_path} is empty. Saving an empty processed file.")
                # Save an empty df with original columns if possible, or just Date
                # df_macro.to_csv(processed_macro_data_path, index=False)
                # For safety, ensure 'Date' column exists if creating from scratch
                if 'Date' not in df_macro.columns:
                    pd.DataFrame(columns=['Date']).to_csv(processed_macro_data_path, index=False)
                else:
                    df_macro.to_csv(processed_macro_data_path, index=False)
                return processed_macro_data_path


            # 1. Date Handling
            if 'Date' not in df_macro.columns:
                print(f"  Error: 'Date' column not found in {raw_macro_data_path}. Cannot process.")
                return None
            try:
                df_macro['Date'] = pd.to_datetime(df_macro['Date']).dt.normalize() # Normalize to midnight
            except Exception as e:
                print(f"  Error parsing 'Date' column in {raw_macro_data_path}: {e}. Cannot process.")
                return None

            df_macro.sort_values(by='Date', inplace=True)

            # 2. Data Cleaning / Validation (Minimal here, as downloader handles ffill/bfill)
            # - Ensure expected columns (from config if available) are present.
            #   The downloader should have produced these.
            if self.config and self.config.get('macro_feature_list'):
                expected_cols = ['Date'] + self.config.get('macro_feature_list')
                missing_cols = [col for col in expected_cols if col not in df_macro.columns]
                if missing_cols:
                    print(f"  Warning: Missing expected macro columns: {missing_cols}. Columns found: {df_macro.columns.tolist()}")

            # - Convert data to numeric, coercing errors (though downloader should provide numeric)
            # for col in df_macro.columns:
            #     if col != 'Date':
            #         df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
            # df_macro = df_macro.ffill().bfill() # Re-apply just in case of new NaNs from coerce

            # 3. Optional: Alignment with a master trading calendar (skipped for now)
            # This would involve reindexing to a trading calendar and ffilling.
            # For example:
            # trading_dates = pd.date_range(start=df_macro['Date'].min(), end=df_macro['Date'].max(), freq=pd.tseries.offsets.BDay())
            # df_macro.set_index('Date', inplace=True)
            # df_macro = df_macro.reindex(trading_dates).ffill()
            # df_macro.reset_index(inplace=True) # 'Date' becomes a column again, named 'index'
            # df_macro.rename(columns={'index': 'Date'}, inplace=True)

            # Save the processed DataFrame
            df_macro.to_csv(processed_macro_data_path, index=False)
            print(f"  Successfully preprocessed macro data and saved to {processed_macro_data_path}")
            return processed_macro_data_path

        except pd.errors.EmptyDataError:
            print(f"  Warning: Raw macro data file {raw_macro_data_path} is empty or contains no data. Skipping.")
            # Create an empty file with Date column
            pd.DataFrame(columns=['Date']).to_csv(processed_macro_data_path, index=False)
            return processed_macro_data_path

        except Exception as e:
            print(f"  Error processing macro data file {raw_macro_data_path}: {e}")
            return None

if __name__ == '__main__':
    print("Running MacroDataProcessor example...")

    # Create dummy config and directories for testing
    _PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    dummy_config_for_testing = {
        'PROJECT_ROOT': _PROJECT_ROOT_TEST,
        'data_dir': 'data_test_macro_processor',
        'macro_data_filename': 'raw_macro_test.csv', # Input for this test
        'processed_macro_filename': 'processed_macro_test.csv', # Output for this test
        'macro_feature_list': ['snp500', 'dgs10']
    }

    # Define test input and output paths
    test_data_base_dir = os.path.join(dummy_config_for_testing['PROJECT_ROOT'], dummy_config_for_testing['data_dir'])
    raw_macro_file = os.path.join(test_data_base_dir, dummy_config_for_testing['macro_data_filename'])
    processed_macro_file = os.path.join(test_data_base_dir, dummy_config_for_testing['processed_macro_filename'])

    # Ensure test directories are clean
    if os.path.exists(test_data_base_dir):
        for f_name in os.listdir(test_data_base_dir):
            if f_name.startswith('raw_macro_test') or f_name.startswith('processed_macro_test'):
                os.remove(os.path.join(test_data_base_dir, f_name))
    else:
        os.makedirs(test_data_base_dir)

    # Create a dummy raw macro CSV file for testing
    dummy_raw_macro_data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-05']),
        'snp500': [3800, None, 3850, 3855], # NaN to test ffill/bfill if re-applied
        'dgs10': [3.5, 3.52, 3.52, None],
        'extra_col': [1,2,3,4] # Extra column not in macro_feature_list
    }
    df_dummy_raw = pd.DataFrame(dummy_raw_macro_data)
    df_dummy_raw.to_csv(raw_macro_file, index=False)
    print(f"Created dummy raw macro file: {raw_macro_file}")

    # Create an empty raw file scenario
    empty_raw_macro_file = os.path.join(test_data_base_dir, "empty_raw_macro.csv")
    open(empty_raw_macro_file, 'w').close()


    processor = MacroDataProcessor(config=dummy_config_for_testing)

    print(f"Test input raw macro file: {raw_macro_file}")
    print(f"Test output processed macro file: {processed_macro_file}")

    # Test with the populated file
    saved_path = processor.run_macro_postProcess(raw_macro_file, processed_macro_file)

    if saved_path and os.path.exists(saved_path):
        print(f"Macro data preprocessing test successful. File saved to: {saved_path}")
        df_check = pd.read_csv(saved_path)
        print(f"Processed data shape: {df_check.shape}")
        print(f"Columns: {df_check.columns.tolist()}")
        print(f"Data (first 3 rows):
{df_check.head(3)}")
        # Check if NaNs are handled (assuming downloader did its job, this processor is light touch)
        # For this test, NaNs should be filled by the downloader's ffill/bfill,
        # this processor mainly ensures format. If it were to re-apply ffill/bfill:
        # if df_check[['snp500', 'dgs10']].isnull().values.any():
        #    print(f"  Warning: Found NaNs in processed macro data for {os.path.basename(saved_path)}.")
    else:
        print(f"Macro data preprocessing test failed for {raw_macro_file} or file not saved.")

    # Test with the empty file
    processed_empty_path = os.path.join(test_data_base_dir, "processed_empty_macro.csv")
    saved_empty_path = processor.run_macro_postProcess(empty_raw_macro_file, processed_empty_path)
    if saved_empty_path and os.path.exists(saved_empty_path):
        print(f"Macro data preprocessing test for empty file successful. File saved to: {saved_empty_path}")
        df_empty_check = pd.read_csv(saved_empty_path)
        print(f"Processed empty data shape: {df_empty_check.shape}") # Should be (0, num_cols) or (0,1) if only Date
        print(f"Processed empty data columns: {df_empty_check.columns.tolist()}")
    else:
        print(f"Macro data preprocessing test for empty file {empty_raw_macro_file} failed.")

    print("MacroDataProcessor example finished.")
