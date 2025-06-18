import os
from src.data_preprocessing.CollectData import StockDataCollector
from src.data_preprocessing.PreProcess import StockDataPreprocessor
from src.data_preprocessing.FeatureEngineer import StockFeatureEngineer
from src.data_preprocessing.Post_Process_Features import StockDataPostProcessor
from src.data_preprocessing.Download_Macro import MacroDataDownloader
from src.data_preprocessing.PreProcessMacro import MacroDataProcessor
from src.data_preprocessing.CombineDf import DataMerger



class StockPreProcessPipeline:
    def __init__(self):
        self.stock_data_dir = 'stock_data'
        self.preprocessed_data_dir = 'preprocessed_data'
        self.feature_added_data_dir = 'feature_added_data'
        self.post_processed_data_dir = 'post_processed_data'
        self.macro_data_file = 'macro_data.csv'
        self.processed_macro_file = 'macro.csv'
        # self.combined_data_file = "data/combined_stock_data.csv" # Now passed to run_data_pipeline

    def run_data_pipeline(self, ticker_list, start_date, end_date, output_csv_path):
        # Step 1: Collect stock data
        collector = StockDataCollector(save_dir=self.stock_data_dir)
        collector.run_collect(ticker_list, start_date, end_date)

        # Step 2: Preprocess stock data
        preprocessor = StockDataPreprocessor()
        preprocessor.run_preprocessing(self.stock_data_dir, self.preprocessed_data_dir)

        # Step 3: Generate features
        feature_engineer = StockFeatureEngineer()
        feature_engineer.run_feature_engineering(
            self.preprocessed_data_dir,
            self.feature_added_data_dir
        )

        # Step 4: Post-process features
        postprocessor = StockDataPostProcessor()
        postprocessor.run_postprocess(
            self.feature_added_data_dir,
            self.post_processed_data_dir
        )

        # Step 5: Download macroeconomic data
        downloader = MacroDataDownloader()
        downloader.run_download_macro(self.macro_data_file)

        # Step 6: Process macro data
        processor = MacroDataProcessor()
        processor.run_macro_postProcess()

        # Step 7: Merge datasets
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # Ensure data directory exists
        merger = DataMerger(
            self.post_processed_data_dir,
            self.processed_macro_file,
            output_csv_path
        )
        combined_file = merger.run_combine_data()
        
        return output_csv_path # Return the path where the file was saved

