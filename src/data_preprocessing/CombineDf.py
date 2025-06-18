# src/data_preprocessing/CombineDf.py
import os
import pandas as pd

class DataMerger:
    def __init__(self, post_processed_stock_dir, processed_macro_path, output_path_for_merger, config=None):
        """
        Initializes the DataMerger.
        :param post_processed_stock_dir: Directory containing post-processed individual stock CSVs.
        :param processed_macro_path: Path to the processed macroeconomic data CSV.
        :param output_path_for_merger: Full path to save the final combined CSV.
        :param config: Configuration dictionary.
        """
        self.post_processed_stock_dir = post_processed_stock_dir
        self.processed_macro_path = processed_macro_path
        self.output_path = output_path_for_merger # Final output path
        self.config = config
        if not self.config:
            print("Warning: DataMerger initialized without a config. Some operations might use defaults or fail.")
            # Provide a minimal config for standalone testing if necessary
            self.config = {
                'PROJECT_ROOT': '.',
                'data_dir': 'data_test_combiner',
                'post_processed_features_dir': 'postprocessed_stock_input', # Input for this test
                'processed_macro_filename': 'processed_macro_input.csv',    # Input for this test
                'final_combined_data_dir': 'final_combined_output',       # Part of output path construction for test
                'final_combined_filename_template': 'test_combined_data.csv' # For test output path
            }

    def run_combine_data(self):
        """
        Loads all post-processed individual stock data and processed macro data,
        merges them into a single DataFrame, and saves it.
        Column names for stocks are suffixed with _0, _1, ...
        The final DataFrame columns must match what MultiStockTradingEnv expects.
        """
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        print(f"[DataMerger] Starting data combination.")
        print(f"Input stock data directory: {self.post_processed_stock_dir}")
        print(f"Input macro data file: {self.processed_macro_path}")
        print(f"Output combined data file: {self.output_path}")

        # 1. Load Macro Data
        df_macro = None
        if os.path.exists(self.processed_macro_path):
            try:
                df_macro = pd.read_csv(self.processed_macro_path)
                if 'Date' not in df_macro.columns:
                    print(f"Error: 'Date' column missing in macro data file: {self.processed_macro_path}. Macro data will not be merged.")
                    df_macro = None
                else:
                    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
                    df_macro.set_index('Date', inplace=True)
                    print(f"  Loaded macro data. Shape: {df_macro.shape}")
            except Exception as e:
                print(f"  Error loading macro data from {self.processed_macro_path}: {e}. Macro data will not be merged.")
                df_macro = None
        else:
            print(f"  Warning: Macro data file not found at {self.processed_macro_path}. Proceeding without macro data.")

        # 2. Load and Combine Individual Stock Data
        all_stock_dfs_to_merge = []
        stock_files = sorted([f for f in os.listdir(self.post_processed_stock_dir) if f.endswith('.csv')]) # Sort for consistent column order

        if not stock_files:
            print("  No stock files found in {self.post_processed_stock_dir}. Cannot create combined stock data.")
            # Depending on requirements, either return None, or proceed with only macro data if it exists.
            # For an RL trading env, no stock data is usually a critical failure.
            if df_macro is not None:
                print("  Only macro data is available. Saving macro data as the final output.")
                df_macro.reset_index(inplace=True) # Ensure 'Date' is a column
                df_macro.to_csv(self.output_path, index=False)
                return self.output_path
            else:
                print("  Error: No stock data and no macro data available. Cannot create output file.")
                return None


        print(f"  Found {len(stock_files)} stock data files to merge.")
        for i, stock_csv in enumerate(stock_files):
            stock_filepath = os.path.join(self.post_processed_stock_dir, stock_csv)
            try:
                df_stock = pd.read_csv(stock_filepath)
                if 'Date' not in df_stock.columns:
                    print(f"    Error: 'Date' column missing in stock file {stock_csv}. Skipping this stock.")
                    continue
                df_stock['Date'] = pd.to_datetime(df_stock['Date'])
                df_stock.set_index('Date', inplace=True)

                # Rename columns: feature_name -> feature_name_i (e.g., close_0, sma50_1)
                # Ensure column names are lowercase before suffixing for consistency.
                # OHLCV were TitleCased by FeatureEngineer, other features might be mixed from pandas_ta.
                # The environment expects specific casings, e.g., 'close_0', 'sma50_0'.
                # So, convert to lower, then add suffix.
                df_stock.columns = [col.lower() for col in df_stock.columns]
                df_stock = df_stock.add_suffix(f'_{i}')

                all_stock_dfs_to_merge.append(df_stock)
                print(f"    Loaded and processed {stock_csv} as stock_{i}. Shape: {df_stock.shape}")
            except Exception as e:
                print(f"    Error loading or processing stock file {stock_csv}: {e}. Skipping.")

        if not all_stock_dfs_to_merge:
            print("  Error: No stock data could be successfully loaded and processed.")
            if df_macro is not None: # Save only macro if it exists
                print("  Saving only macro data as no stock data was processed.")
                df_macro.reset_index(inplace=True)
                df_macro.to_csv(self.output_path, index=False)
                return self.output_path
            return None # Critical failure

        df_combined_stocks = pd.concat(all_stock_dfs_to_merge, axis=1)
        print(f"  Combined all stock data. Shape: {df_combined_stocks.shape}")

        # 3. Merge Stocks with Macro Data
        if df_macro is not None:
            # Outer join to keep all dates from both stock and macro data
            final_df = pd.merge(df_combined_stocks, df_macro, left_index=True, right_index=True, how='outer')
            print(f"  Merged stock data with macro data. Shape after merge: {final_df.shape}")
        else:
            final_df = df_combined_stocks
            print("  No macro data to merge. Using combined stock data as final.")

        # 4. Post-Merge Cleaning
        if final_df.empty:
            print("  Error: Final DataFrame is empty after merging. Cannot save.")
            return None

        final_df.sort_index(inplace=True) # Sort by Date index

        # Fill NaNs that may have resulted from outer join or if individual series had them
        # Example: macro data might be less frequent than stock data, or vice-versa for specific dates.
        final_df = final_df.ffill().bfill()
        print("  Applied ffill and bfill to the final merged DataFrame.")

        # Optional: Drop rows where critical data might still be missing.
        # E.g., if 'close_0' is NaN, that row is likely unusable.
        # This depends on how many stocks are primary vs auxiliary.
        # For now, assume ffill/bfill is sufficient. If not, add:
        # primary_close_col = 'close_0' # Assuming first stock is primary
        # if primary_close_col in final_df.columns:
        #     final_df.dropna(subset=[primary_close_col], inplace=True)
        #     print(f"  Dropped rows where '{primary_close_col}' is NaN. New shape: {final_df.shape}")

        final_df.reset_index(inplace=True) # Move 'Date' back to a column
        # Ensure 'Date' column is named 'Date' after reset_index if index had no name
        if 'index' in final_df.columns and 'Date' not in final_df.columns:
            final_df.rename(columns={'index': 'Date'}, inplace=True)


        # Final check for columns expected by MultiStockTradingEnv's _validate_data
        # This is important. The environment expects specific column names.
        # E.g., 'close_0', 'sma50_0', 'eps_0', 'pe_ratio_0', 'volatility_30d_0', 'momentum_0', 'volume_trend_0'
        # And macro features like 'snp500', 'gold_price' (without suffixes).
        # The renaming in the stock loop (e.g. df_stock.add_suffix(f'_{i}')) should handle the _i part.
        # Make sure feature names from FeatureEngineer (e.g. 'SMA_50', 'RSI_14') become 'sma_50_i', 'rsi_14_i'.
        # Current FeatureEngineer saves pandas_ta default names (e.g. 'SMA_50').
        # The add_suffix here makes it 'sma_50_i' (after lowercasing). This should align with env.

        try:
            final_df.to_csv(self.output_path, index=False)
            print(f"  Successfully saved final combined data to {self.output_path}. Final shape: {final_df.shape}")
            return self.output_path
        except Exception as e:
            print(f"  Error saving final combined data to {self.output_path}: {e}")
            return None


if __name__ == '__main__':
    print("Running DataMerger example...")

    # Setup dummy config and directories for testing
    _PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    dummy_combiner_config = {
        'PROJECT_ROOT': _PROJECT_ROOT_TEST,
        'data_dir': 'data_test_combiner', # Base for test subdirs
        'post_processed_features_dir': 'postprocessed_stock_input_combiner', # Input stock data
        'processed_macro_filename': 'processed_macro_input_combiner.csv',    # Input macro data
        # Output path for the merged file will be constructed from these + a filename
        'final_combined_data_dir': 'final_combined_output_combiner',
        'final_combined_filename_template': 'test_combined_data_example.csv'
    }

    test_base_dir = os.path.join(dummy_combiner_config['PROJECT_ROOT'], dummy_combiner_config['data_dir'])

    # Input directories/files for the test
    test_postprocessed_stock_dir = os.path.join(test_base_dir, dummy_combiner_config['post_processed_features_dir'])
    test_processed_macro_path = os.path.join(test_base_dir, dummy_combiner_config['processed_macro_filename'])

    # Output path for the test
    test_output_merged_dir = os.path.join(test_base_dir, dummy_combiner_config['final_combined_data_dir'])
    test_output_merged_file = os.path.join(test_output_merged_dir, dummy_combiner_config['final_combined_filename_template'])


    for dir_path in [test_postprocessed_stock_dir, test_output_merged_dir, os.path.dirname(test_processed_macro_path)]:
        if os.path.exists(dir_path):
            # Clean specific test files, not necessarily the whole directory if shared
            for f_name in os.listdir(dir_path):
                 if f_name.startswith("STOCK") or f_name == os.path.basename(test_processed_macro_path) or f_name == os.path.basename(test_output_merged_file):
                    try: os.remove(os.path.join(dir_path, f_name))
                    except OSError: pass # if it's a dir
        else:
            os.makedirs(dir_path, exist_ok=True)

    # Create dummy post-processed stock CSVs
    dates1 = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 6)])
    stock1_data = pd.DataFrame({
        'Date': dates1, 'Open': np.arange(10,15), 'Close': np.arange(10.5,15.5), 'SMA_50': np.arange(9,14), 'Volume': np.arange(100,105)
    })
    stock1_data.to_csv(os.path.join(test_postprocessed_stock_dir, 'STOCKA.csv'), index=False)

    dates2 = pd.to_datetime([f'2023-01-{i:02d}' for i in range(3, 8)]) # Different date range
    stock2_data = pd.DataFrame({
        'Date': dates2, 'Open': np.arange(20,25), 'Close': np.arange(20.5,25.5), 'RSI_14': np.arange(50,55), 'Volume': np.arange(200,205)
    })
    stock2_data.to_csv(os.path.join(test_postprocessed_stock_dir, 'STOCKB.csv'), index=False)
    print(f"Created dummy stock files in: {test_postprocessed_stock_dir}")

    # Create dummy processed macro CSV
    macro_dates = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 10)]) # Wider range
    macro_data = pd.DataFrame({
        'Date': macro_dates, 'snp500': np.arange(3800, 3809), 'dgs10': np.linspace(3.5, 3.8, 9)
    })
    macro_data.to_csv(test_processed_macro_path, index=False)
    print(f"Created dummy macro file: {test_processed_macro_path}")


    merger = DataMerger(
        post_processed_stock_dir=test_postprocessed_stock_dir,
        processed_macro_path=test_processed_macro_path,
        output_path_for_merger=test_output_merged_file,
        config=dummy_combiner_config
    )

    saved_final_path = merger.run_combine_data()

    if saved_final_path and os.path.exists(saved_final_path):
        print(f"Data combination test successful. Final merged file: {saved_final_path}")
        df_final_check = pd.read_csv(saved_final_path)
        print(f"Final data shape: {df_final_check.shape}")
        print(f"Final columns: {df_final_check.columns.tolist()}")
        print(f"Final data (first 5 rows):
{df_final_check.head()}")
        # Verify date range (should be union of all inputs)
        min_date = pd.to_datetime(df_final_check['Date']).min()
        max_date = pd.to_datetime(df_final_check['Date']).max()
        print(f"Date range in final data: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        # Check for NaNs (should be handled by ffill/bfill)
        if df_final_check.isnull().values.any():
            print(f"  Warning: Found NaNs in final combined data:
{df_final_check.isnull().sum()}")
        else:
            print("  No NaNs found in final combined data.")
    else:
        print("Data combination test failed or file not saved.")

    print("DataMerger example finished.")
