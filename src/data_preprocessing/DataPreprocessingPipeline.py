# src/data_preprocessing/DataPreprocessingPipeline.py
import os
# Import all the implemented component classes
from .CollectData import StockDataCollector
from .PreProcess import StockDataPreprocessor
from .FeatureEngineer import StockFeatureEngineer
from .Post_Process_Features import StockDataPostProcessor
from .Download_Macro import MacroDataDownloader
from .PreProcessMacro import MacroDataProcessor
from .CombineDf import DataMerger

class StockPreProcessPipeline:
    def __init__(self, config=None):
        """
        Initialize the pipeline and its components.
        Relies on the provided config for paths and parameters.
        :param config: Loaded configuration dictionary.
        """
        if not config:
            raise ValueError("StockPreProcessPipeline requires a configuration object.")
        self.config = config

        # Determine PROJECT_ROOT from config (should be added by config_loader)
        self.project_root = self.config.get('PROJECT_ROOT', '.') # Default to current if not in config

        # Define base data directory from config, relative to project_root
        self.data_dir = os.path.join(self.project_root, self.config.get('data_dir', 'data'))

        # Define directories for each stage of the pipeline, relative to self.data_dir
        self.raw_stock_dir = os.path.join(self.data_dir, self.config.get('raw_stock_data_dir', 'raw_stock_data'))
        self.preprocessed_stock_dir = os.path.join(self.data_dir, self.config.get('processed_stock_data_dir', 'processed_stock_data'))
        self.feature_added_dir = os.path.join(self.data_dir, self.config.get('feature_engineered_data_dir', 'feature_engineered_data'))
        # Directory for individual stock files after their final post-processing, before merging
        self.post_processed_individual_stock_dir = os.path.join(self.data_dir, self.config.get('post_processed_features_dir', 'post_processed_individual_stock_data'))

        # Paths for macro data files within self.data_dir
        self.macro_data_file = os.path.join(self.data_dir, self.config.get('macro_data_filename', 'macro_economic_data.csv'))
        self.processed_macro_file = os.path.join(self.data_dir, self.config.get('processed_macro_filename', 'processed_macro_economic_data.csv'))

        # Instantiate components, passing the config to each
        self.collector = StockDataCollector(config=self.config) # It will derive its save_dir from config
        self.preprocessor = StockDataPreprocessor(config=self.config)
        self.feature_engineer = StockFeatureEngineer(config=self.config)
        self.post_processor = StockDataPostProcessor(config=self.config)
        self.macro_downloader = MacroDataDownloader(config=self.config)
        self.macro_processor = MacroDataProcessor(config=self.config)
        # DataMerger is instantiated in run_data_pipeline with specific I/O paths

    def run_data_pipeline(self, ticker_list, start_date_str, end_date_str, output_path_for_merger):
        """
        Runs the full data preprocessing pipeline using the implemented components.
        :param ticker_list: List of stock tickers.
        :param start_date_str: Start date for data collection (YYYY-MM-DD).
        :param end_date_str: End date for data collection (YYYY-MM-DD).
        :param output_path_for_merger: The final absolute path for the combined CSV.
        :return: Path to the final processed and combined CSV file, or None if failed.
        """
        print(f"--- Running Data Preprocessing Pipeline ---")
        print(f"Tickers: {ticker_list}, Start: {start_date_str}, End: {end_date_str}")
        print(f"Final output for merged data expected at: {output_path_for_merger}")

        # Ensure all necessary directories for pipeline stages exist
        # Components themselves also create their specific output dirs, but good to have them here.
        dirs_to_create = [
            self.raw_stock_dir, self.preprocessed_stock_dir,
            self.feature_added_dir, self.post_processed_individual_stock_dir,
            os.path.dirname(self.macro_data_file),
            os.path.dirname(self.processed_macro_file),
            os.path.dirname(output_path_for_merger) # Dir for the final merged file
        ]
        for d in dirs_to_create:
            if d and not os.path.exists(d): # Check if d is not empty (e.g. if filename is at root)
                os.makedirs(d, exist_ok=True)
                print(f"Ensured directory exists: {d}")

        # --- Stage 1: Collect Stock Data ---
        print("
[Pipeline Stage 1: Collecting Stock Data]")
        # StockDataCollector saves to self.raw_stock_dir (derived from its config)
        collected_tickers = self.collector.run_collect(ticker_list, start_date_str, end_date_str)
        if not collected_tickers:
            print("Pipeline Error: Stock data collection failed or yielded no tickers. Aborting.")
            return None
        # Update ticker_list to only those successfully collected if needed, or fail if any failed.
        # For now, assume if some failed, pipeline might still proceed with subset or user handles it.

        # --- Stage 2: Preprocess Stock Data ---
        print("
[Pipeline Stage 2: Preprocessing Stock Data]")
        preprocessed_stock_files = self.preprocessor.run_preprocessing(self.raw_stock_dir, self.preprocessed_stock_dir)
        if not preprocessed_stock_files:
            print("Pipeline Error: Stock data preprocessing failed. Aborting.")
            return None

        # --- Stage 3: Feature Engineer Stock Data ---
        print("
[Pipeline Stage 3: Engineering Features for Stock Data]")
        feature_engineered_files = self.feature_engineer.run_feature_engineering(self.preprocessed_stock_dir, self.feature_added_dir)
        if not feature_engineered_files:
            print("Pipeline Error: Stock feature engineering failed. Aborting.")
            return None

        # --- Stage 4: Post-Process Stock Features ---
        print("
[Pipeline Stage 4: Post-Processing Stock Features]")
        post_processed_stock_files = self.post_processor.run_postprocess(self.feature_added_dir, self.post_processed_individual_stock_dir)
        if not post_processed_stock_files:
            print("Pipeline Error: Stock feature post-processing failed. Aborting.")
            return None

        # --- Stage 5: Download Macro Data ---
        print("
[Pipeline Stage 5: Downloading Macroeconomic Data]")
        # macro_data_file path is absolute, derived in __init__
        downloaded_macro_path = self.macro_downloader.run_download_macro(self.macro_data_file, start_date_str=start_date_str, end_date_str=end_date_str)
        if not downloaded_macro_path:
            print("Pipeline Warning: Macro data download failed. Proceeding without macro data if CombineDf handles it.")
            # If macro data is essential, one might choose to abort here.
            # For now, let DataMerger handle a missing macro file.

        # --- Stage 6: Preprocess Macro Data ---
        print("
[Pipeline Stage 6: Preprocessing Macroeconomic Data]")
        # processed_macro_file path is absolute, derived in __init__
        # Only run if download was successful (or if dummy file exists at self.macro_data_file)
        processed_macro_data_final_path = None
        if os.path.exists(self.macro_data_file): # Check if raw macro file exists
             processed_macro_data_final_path = self.macro_processor.run_macro_postProcess(self.macro_data_file, self.processed_macro_file)
             if not processed_macro_data_final_path:
                 print("Pipeline Warning: Macro data preprocessing failed. Proceeding without preprocessed macro data.")
        else:
            print("Pipeline Info: Raw macro data file not found, skipping macro preprocessing.")


        # --- Stage 7: Combine Stock and Macro Data ---
        print("
[Pipeline Stage 7: Combining Stock and Macro Data]")
        # DataMerger needs the path to directory of individual post-processed stock files,
        # and the path to the single processed macro data CSV file.
        # output_path_for_merger is the final destination for the combined data.
        merger = DataMerger(
            post_processed_stock_dir=self.post_processed_individual_stock_dir,
            processed_macro_path=self.processed_macro_file if processed_macro_data_final_path else None, # Pass None if macro processing failed
            output_path_for_merger=output_path_for_merger,
            config=self.config
        )
        final_combined_data_path = merger.run_combine_data()

        if not final_combined_data_path:
            print("Pipeline Error: Final data combination failed. Aborting.")
            return None

        print(f"
--- Data Preprocessing Pipeline Finished Successfully ---")
        print(f"Final combined data saved to: {final_combined_data_path}")
        return final_combined_data_path

if __name__ == '__main__':
    from src.utils.config_loader import get_config # For testing
    from datetime import datetime, timedelta # Added for example usage

    print("Running StockPreProcessPipeline example...")
    try:
        # Load main config (assuming shared_config.yaml is in ../../config from this file's location)
        # This requires config_loader.py to be in src/utils/
        # Adjust path to config if running this test from a different working directory.
        # For this test, let's assume config_loader can find shared_config.yaml via its mechanisms.
        
        # Construct path to config assuming this script is in src/data_preprocessing
        # and shared_config.yaml is in config/ at project root
        _config_path_for_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'shared_config.yaml'))
        if not os.path.exists(_config_path_for_test):
             print(f"Test Error: shared_config.yaml not found at expected path for testing: {_config_path_for_test}")
             print("Attempting to load config from default path expected by get_config (config/shared_config.yaml relative to project root).")
             config = get_config() # Try default loader path
        else:
            # If we want to force get_config to use a specific path for testing:
            # This assumes get_config can accept a path, or we temporarily modify how it finds its path.
            # For simplicity, if this __main__ is run, ensure your CWD or its logic finds the config.
            # Or, provide a dummy config here if get_config() fails.
            print(f"Attempting to load config from: {_config_path_for_test}")
            config = get_config() # Assuming get_config knows its default relative path from project root

        if not config:
            print("Test Error: Could not load shared_config.yaml for pipeline test. Ensure it exists and paths are correct.")
            exit()

        # Override specific paths in config for isolated testing if needed, or ensure config is set for testing
        # For this example, we'll use paths from the loaded config.
        # Ensure PROJECT_ROOT is in config from loader.
        project_root = config.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

        # Define a specific output directory for this test run's final merged file
        test_output_dir_base = os.path.join(project_root, config.get('data_dir', 'data'), "pipeline_test_output")
        os.makedirs(test_output_dir_base, exist_ok=True)

        # Clean up previous test output directory (optional)
        # for f in os.listdir(test_output_dir_base): os.remove(os.path.join(test_output_dir_base, f))
        # For intermediate dirs, the pipeline components should handle their own test cleanup or use unique dirs.
        # This test focuses on the pipeline's orchestration.

        pipeline = StockPreProcessPipeline(config=config)

        # Use a small list of tickers and a short date range for quick testing
        test_tickers = config.get('default_ticker_list', ["AAPL", "MSFT"])[:2] # Max 2 tickers for test
        test_start = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d') # ~4 months of data
        test_end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Define where the final merged file for this test run should go
        final_merged_file_for_test = os.path.join(test_output_dir_base, f"test_run_merged_{'_'.join(test_tickers)}_{test_start}_{test_end}.csv")

        print(f"Pipeline test will use tickers: {test_tickers}")
        print(f"Date range: {test_start} to {test_end}")
        print(f"Test output for final merged file: {final_merged_file_for_test}")

        # Ensure intermediate directories specified in config are writable or adjust config for test
        # Example: 'raw_stock_data_dir' should be a place where StockDataCollector can write.
        # The pipeline __init__ now uses these from config to set up its stage-specific paths.

        result_path = pipeline.run_data_pipeline(test_tickers, test_start, test_end, final_merged_file_for_test)

        if result_path and os.path.exists(result_path):
            print(f"
Pipeline example run successful! Final data at: {result_path}")
            # df_final = pd.read_csv(result_path)
            # print(f"Final DataFrame shape: {df_final.shape}")
            # print(f"Final DataFrame columns: {df_final.columns.tolist()}")
            # print(f"Final DataFrame head:
{df_final.head()}")
        else:
            print("
Pipeline example run failed or did not produce an output file.")

    except FileNotFoundError as e:
        # This might occur if shared_config.yaml is not found by get_config()
        print(f"Test Error: A file was not found. This might be shared_config.yaml or an issue with paths. Details: {e}")
    except Exception as e:
        print(f"An error occurred during the pipeline example: {e}")
        import traceback
        traceback.print_exc()
