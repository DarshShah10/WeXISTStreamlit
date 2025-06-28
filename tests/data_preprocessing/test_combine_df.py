import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime

# Adjust import path based on your project structure
from src.data_preprocessing.CombineDf import DataMerger

class TestDataMerger(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.stock_input_dir = os.path.join(self.test_dir, "postprocessed_stock_data")
        self.macro_input_dir = os.path.join(self.test_dir, "macro_data") # Dir for macro file
        self.output_dir = os.path.join(self.test_dir, "final_combined_data")

        os.makedirs(self.stock_input_dir, exist_ok=True)
        os.makedirs(self.macro_input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.dummy_macro_path = os.path.join(self.macro_input_dir, "processed_macro.csv")
        self.final_output_path = os.path.join(self.output_dir, "merged_data.csv")

        # Minimal config for DataMerger if it uses it (currently it doesn't directly use much from config for its core logic)
        self.dummy_config = {
            # PROJECT_ROOT, data_dir etc. are not directly used by DataMerger's run_combine_data logic
            # as it operates on paths passed to its constructor.
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_stock_csv(self, filename, start_date_str, num_days, data_prefix=""):
        filepath = os.path.join(self.stock_input_dir, filename)
        dates = pd.to_datetime([pd.to_datetime(start_date_str) + pd.Timedelta(days=i) for i in range(num_days)])
        data = {
            'Date': dates,
            # FeatureEngineer produces TitleCase OHLCV, and other features might be specific (e.g., SMA_50)
            'Open': np.random.uniform(10, 12, size=num_days),
            'Close': np.random.uniform(10, 12, size=num_days),
            'SMA_50': np.random.uniform(9, 11, size=num_days),
            'RSI_14': np.random.uniform(40, 60, size=num_days)
        }
        pd.DataFrame(data).to_csv(filepath, index=False)
        return filepath

    def _create_dummy_macro_csv(self, start_date_str, num_days):
        dates = pd.to_datetime([pd.to_datetime(start_date_str) + pd.Timedelta(days=i) for i in range(num_days)])
        data = {
            'Date': dates,
            'snp500': np.random.uniform(3000, 4000, size=num_days),
            'dgs10': np.random.uniform(1.5, 2.5, size=num_days)
        }
        pd.DataFrame(data).to_csv(self.dummy_macro_path, index=False)
        return self.dummy_macro_path

    def test_merge_with_stock_and_macro_data(self):
        self._create_dummy_stock_csv("STOCKA.csv", "2023-01-01", 10)
        self._create_dummy_stock_csv("STOCKB.csv", "2023-01-03", 10) # Overlapping but different start
        self._create_dummy_macro_csv("2023-01-01", 15)

        merger = DataMerger(self.stock_input_dir, self.dummy_macro_path, self.final_output_path, config=self.dummy_config)
        result_path = merger.run_combine_data()

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))

        df_merged = pd.read_csv(result_path)
        self.assertIn('Date', df_merged.columns)
        self.assertIn('close_0', df_merged.columns) # DataMerger lowercases and suffixes
        self.assertIn('sma_50_0', df_merged.columns)
        self.assertIn('close_1', df_merged.columns)
        self.assertIn('rsi_14_1', df_merged.columns)
        self.assertIn('snp500', df_merged.columns) # Macro columns are not suffixed
        self.assertIn('dgs10', df_merged.columns)

        # Check for NaN handling (outer join + ffill/bfill should handle most)
        self.assertFalse(df_merged.isnull().values.any(), "No NaNs should be present in final merged data.")

        # Check date range (should be union of all inputs)
        # Stock A: 2023-01-01 to 2023-01-10
        # Stock B: 2023-01-03 to 2023-01-12
        # Macro:   2023-01-01 to 2023-01-15
        # Expected: 2023-01-01 to 2023-01-15 (if all data filled)
        self.assertEqual(pd.to_datetime(df_merged['Date'].min()).strftime('%Y-%m-%d'), "2023-01-01")
        self.assertEqual(pd.to_datetime(df_merged['Date'].max()).strftime('%Y-%m-%d'), "2023-01-15")


    def test_merge_only_stock_data(self):
        self._create_dummy_stock_csv("STOCKA.csv", "2023-01-01", 5)
        # No macro data file created

        merger = DataMerger(self.stock_input_dir, "non_existent_macro.csv", self.final_output_path, config=self.dummy_config)
        result_path = merger.run_combine_data()

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        df_merged = pd.read_csv(result_path)

        self.assertIn('close_0', df_merged.columns)
        self.assertNotIn('snp500', df_merged.columns)
        self.assertFalse(df_merged.isnull().values.any())

    def test_merge_only_macro_data(self):
        # No stock data files created in self.stock_input_dir
        self._create_dummy_macro_csv("2023-01-01", 5)

        merger = DataMerger(self.stock_input_dir, self.dummy_macro_path, self.final_output_path, config=self.dummy_config)
        result_path = merger.run_combine_data()

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        df_merged = pd.read_csv(result_path)

        self.assertIn('snp500', df_merged.columns)
        self.assertNotIn('close_0', df_merged.columns)
        self.assertFalse(df_merged.isnull().values.any())

    def test_merge_no_data_at_all(self):
        # No stock files, no macro file
        merger = DataMerger(self.stock_input_dir, "non_existent_macro.csv", self.final_output_path, config=self.dummy_config)
        result_path = merger.run_combine_data()
        self.assertIsNone(result_path, "Should return None if no data at all.")
        self.assertFalse(os.path.exists(self.final_output_path)) # No file should be created

    def test_stock_file_missing_date_column(self):
        # Create one valid stock file
        self._create_dummy_stock_csv("STOCKA.csv", "2023-01-01", 5)
        # Create one stock file missing 'Date'
        bad_stock_path = os.path.join(self.stock_input_dir, "STOCKB_NODATE.csv")
        pd.DataFrame({'Close': [1,2,3]}).to_csv(bad_stock_path, index=False)

        self._create_dummy_macro_csv("2023-01-01", 5) # Add macro data

        merger = DataMerger(self.stock_input_dir, self.dummy_macro_path, self.final_output_path, config=self.dummy_config)
        result_path = merger.run_combine_data()

        self.assertTrue(os.path.exists(result_path))
        df_merged = pd.read_csv(result_path)

        self.assertIn('close_0', df_merged.columns) # STOCKA should be there (index 0)
        self.assertNotIn('close_1', df_merged.columns) # STOCKB_NODATE should be skipped
        self.assertIn('snp500', df_merged.columns)

    def test_macro_file_missing_date_column(self):
        self._create_dummy_stock_csv("STOCKA.csv", "2023-01-01", 5)
        # Create macro file missing 'Date'
        bad_macro_path = os.path.join(self.macro_input_dir, "macro_nodate.csv")
        pd.DataFrame({'snp500': [3000, 3001]}).to_csv(bad_macro_path, index=False)

        merger = DataMerger(self.stock_input_dir, bad_macro_path, self.final_output_path, config=self.dummy_config)
        result_path = merger.run_combine_data()

        self.assertTrue(os.path.exists(result_path))
        df_merged = pd.read_csv(result_path)

        self.assertIn('close_0', df_merged.columns)
        self.assertNotIn('snp500', df_merged.columns) # Macro data should not be merged

if __name__ == '__main__':
    unittest.main()
