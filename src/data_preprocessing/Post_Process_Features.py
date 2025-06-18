# src/data_preprocessing/Post_Process_Features.py
import os
import pandas as pd

class StockDataPostProcessor:
    def __init__(self, config=None):
        """
        Initializes the StockDataPostProcessor.
        :param config: Configuration dictionary (currently not heavily used by this class,
                       but good practice for consistency and future needs like specific column lists).
        """
        self.config = config
        # This class primarily operates on paths provided to its methods.

    def run_postprocess(self, feature_added_data_dir, post_processed_data_dir):
        """
        Loads feature-engineered stock data, performs final post-processing,
        and saves the data. For now, this is a light-touch step, mainly ensuring
        no NaNs remain and data is in good order.

        :param feature_added_data_dir: Directory containing feature-engineered stock CSV files.
        :param post_processed_data_dir: Directory to save post-processed CSV files.
        :return: List of successfully post-processed file paths.
        """
        if not os.path.exists(feature_added_data_dir):
            print(f"Error: Input directory for post-processing does not exist: {feature_added_data_dir}")
            return []

        if not os.path.exists(post_processed_data_dir):
            os.makedirs(post_processed_data_dir, exist_ok=True)
            print(f"Created directory: {post_processed_data_dir}")

        processed_files_list = []
        input_csv_files = [f for f in os.listdir(feature_added_data_dir) if f.endswith('.csv')]

        if not input_csv_files:
            print(f"No CSV files found in {feature_added_data_dir}. Nothing to post-process.")
            return []

        print(f"[StockDataPostProcessor] Starting post-processing for {len(input_csv_files)} files from {feature_added_data_dir}.")
        print(f"Output directory: {post_processed_data_dir}")

        for csv_file in input_csv_files:
            input_filepath = os.path.join(feature_added_data_dir, csv_file)
            output_filepath = os.path.join(post_processed_data_dir, csv_file)

            print(f"  Post-processing file: {csv_file}...")
            try:
                df = pd.read_csv(input_filepath)

                if df.empty:
                    print(f"    Warning: Feature-engineered file {csv_file} is empty. Skipping.")
                    # Optionally save an empty file or log this.
                    continue

                # 1. Date Handling (ensure it's present and parsed, though FeatureEngineer should ensure this)
                if 'Date' not in df.columns:
                    print(f"    Error: 'Date' column not found in {csv_file}. Skipping post-processing.")
                    continue
                try:
                    # Ensure Date is datetime, but don't set as index unless needed for an operation
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    print(f"    Error parsing 'Date' column in {csv_file} during post-processing: {e}. Skipping.")
                    continue

                # 2. Final NaN Check and Fill
                # FeatureEngineer already does ffill/bfill. This is a final safeguard.
                # It's important if some features might still have NaNs due to very short data series
                # or extreme lookback periods relative to data length.
                # df.set_index('Date', inplace=True) # If operations need Date as index
                # df.sort_index(inplace=True) # Ensure chronological before ffill/bfill

                # Record columns with all NaNs before fill, as they might indicate issues
                cols_all_nan_before = df.columns[df.isnull().all()].tolist()
                if cols_all_nan_before:
                    print(f"    Warning: Columns in {csv_file} with all NaN values before final fill: {cols_all_nan_before}")

                df = df.ffill().bfill()

                # df.reset_index(inplace=True) # If Date was set as index

                # 3. Check if critical columns (e.g., 'Close') became all NaN (highly unlikely if data existed)
                if df['Close'].isnull().all():
                    print(f"    Error: 'Close' column in {csv_file} is all NaN after final fill. Skipping output.")
                    continue

                if df.empty: # Should not happen if Close is not all NaN
                    print(f"    Warning: DataFrame for {csv_file} became empty during post-processing. Skipping output.")
                    continue

                # 4. Optional: Column Selection/Reordering (Not implemented in this version)
                # If self.config had a 'final_feature_columns' list, one could do:
                # desired_columns = self.config.get('final_feature_columns', df.columns.tolist())
                # df = df[[col for col in desired_columns if col in df.columns]]

                # 5. Optional: Outlier Handling (Not implemented in this version)

                # Save the post-processed DataFrame
                df.to_csv(output_filepath, index=False)
                print(f"  Successfully post-processed and saved {csv_file} to {output_filepath}")
                processed_files_list.append(output_filepath)

            except pd.errors.EmptyDataError:
                print(f"    Warning: File {csv_file} is empty or contains no data. Skipping post-processing.")
            except Exception as e:
                print(f"    Error post-processing file {csv_file}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[StockDataPostProcessor] Finished post-processing. Successfully processed {len(processed_files_list)}/{len(input_csv_files)} files.")
        return processed_files_list

if __name__ == '__main__':
    print("Running StockDataPostProcessor example...")

    # Setup dummy config and directories for testing
    _PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    dummy_post_config = {
        'PROJECT_ROOT': _PROJECT_ROOT_TEST,
        'data_dir': 'data_test_postprocessor',
        'feature_engineered_data_dir': 'fe_input_for_postp', # Input for this test
        'post_processed_features_dir': 'postprocessed_output_postp' # Output for this test
        # 'final_feature_columns': ['Date', 'Close', 'SMA_50', 'RSI_14'] # Example for column selection
    }

    test_postp_base_dir = os.path.join(dummy_post_config['PROJECT_ROOT'], dummy_post_config['data_dir'])
    test_postp_input_dir = os.path.join(test_postp_base_dir, dummy_post_config['feature_engineered_data_dir'])
    test_postp_output_dir = os.path.join(test_postp_base_dir, dummy_post_config['post_processed_features_dir'])

    for dir_path in [test_postp_input_dir, test_postp_output_dir]:
        if os.path.exists(dir_path):
            for f_name in os.listdir(dir_path): os.remove(os.path.join(dir_path, f_name))
        else: os.makedirs(dir_path)

    # Create a dummy feature-engineered CSV file
    dates = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 6)])
    dummy_fe_data = {
        'Date': dates,
        'Open': [10, 11, 12, 13, 14],
        'Close': [10.5, 11.5, 12.5, 13.5, 14.5],
        'SMA_50': [None, None, 10.2, 11.0, 11.8], # Leading NaNs
        'RSI_14': [None, 50.0, 55.0, None, 60.0], # Mixed NaNs
        'Volume': [100,200,300,400,500]
    }
    pd.DataFrame(dummy_fe_data).to_csv(os.path.join(test_postp_input_dir, 'TEST_POSTP.csv'), index=False)
    print(f"Created dummy feature-engineered file: TEST_POSTP.csv in {test_postp_input_dir}")

    # Create an empty feature-engineered file
    open(os.path.join(test_postp_input_dir, 'EMPTY_FE.csv'), 'w').close()


    postprocessor = StockDataPostProcessor(config=dummy_post_config)

    print(f"Test input directory (feature-engineered): {test_postp_input_dir}")
    print(f"Test output directory (post-processed): {test_postp_output_dir}")

    postprocessed_files = postprocessor.run_postprocess(test_postp_input_dir, test_postp_output_dir)
    print(f"Files successfully post-processed in test: {postprocessed_files}")

    for file_path in postprocessed_files:
        if os.path.exists(file_path):
            print(f"Verified: File '{file_path}' exists.")
            df_check = pd.read_csv(file_path)
            print(f"Data for {os.path.basename(file_path)} (first 3 rows):
{df_check.head(3)}")
            if df_check.isnull().values.any():
                 print(f"  Warning: Found NaNs in {os.path.basename(file_path)} after post-processing:
{df_check.isnull().sum()}")
            else:
                 print(f"  No NaNs found in {os.path.basename(file_path)} after post-processing.")
        else:
            print(f"Error: Post-processed file '{file_path}' NOT found.")

    print("StockDataPostProcessor example finished.")
