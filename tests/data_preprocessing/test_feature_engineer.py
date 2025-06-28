import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Adjust import path based on your project structure
from src.data_preprocessing.FeatureEngineer import StockFeatureEngineer
# Need yfinance to mock it, but not for actual downloads in test
import yfinance as yf

class TestStockFeatureEngineer(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "preprocessed_data")
        self.output_dir = os.path.join(self.test_dir, "feature_engineered_data")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Minimal config for testing
        self.dummy_config = {
            'tech_indicator_list': ['sma50', 'rsi', 'macd', 'bbands'],
            'fundamental_features': {
                'volatility_window': 5, # Small window for test data
                'momentum_window': 3,
                'volume_trend_window': 2
            },
            'default_eps': 1.0,
            'default_pe_ratio': 15.0
            # PROJECT_ROOT, data_dir etc. are not directly used by FeatureEngineer methods
            # as it operates on paths passed to run_feature_engineering
        }
        self.engineer = StockFeatureEngineer(config=self.dummy_config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_csv(self, filename="TEST.csv", rows=60, include_all_cols=True):
        filepath = os.path.join(self.input_dir, filename)
        dates = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, rows + 1)])
        data = {
            'Date': dates,
            'Open': np.random.uniform(100, 102, size=rows),
            'High': np.random.uniform(102, 104, size=rows),
            'Low': np.random.uniform(98, 100, size=rows),
            'Close': np.random.uniform(100, 103, size=rows),
            'Volume': np.random.randint(10000, 20000, size=rows)
        }
        if not include_all_cols:
            data.pop('Volume') # Example of missing column

        df = pd.DataFrame(data)
        df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
        df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))
        df.to_csv(filepath, index=False)
        return filepath

    def test_empty_input_dir(self):
        results = self.engineer.run_feature_engineering(self.input_dir, self.output_dir)
        self.assertEqual(len(results), 0)

    def test_empty_input_csv(self):
        empty_file = os.path.join(self.input_dir, "EMPTY.csv")
        with open(empty_file, 'w') as f:
            f.write("Date,Open,High,Low,Close,Volume
") # Header only
        results = self.engineer.run_feature_engineering(self.input_dir, self.output_dir)
        self.assertEqual(len(results), 0) # Should skip the empty file

    def test_missing_date_column(self):
        filepath = os.path.join(self.input_dir, "NO_DATE.csv")
        pd.DataFrame({'Open': [1], 'Close': [1]}).to_csv(filepath, index=False)
        results = self.engineer.run_feature_engineering(self.input_dir, self.output_dir)
        self.assertEqual(len(results), 0)

    @patch('yfinance.Ticker')
    def test_full_processing_with_mocked_yf(self, MockTicker):
        # Setup mock for yf.Ticker().info
        mock_instance = MockTicker.return_value
        mock_instance.info = {
            'trailingEps': 2.5,
            'trailingPE': 20.0,
            # Add other keys from self.engineer.features if they are fetched from .info
            "marketCap": 2e12, "pegRatio": 1.5, "priceToBook": 5.0,
            "forwardEps": 2.8, "returnOnEquity": 0.15, "debtToEquity": 0.5,
            "dividendYield": 0.015, "beta": 1.1, "revenueGrowth": 0.1, "earningsGrowth": 0.12
        }

        self._create_dummy_csv("AAPL.csv")
        results = self.engineer.run_feature_engineering(self.input_dir, self.output_dir)
        self.assertEqual(len(results), 1)

        output_df = pd.read_csv(results[0])

        # Check for TA columns (names depend on pandas-ta defaults and FeatureEngineer logic)
        # Assuming FeatureEngineer maps config names to pandas-ta calls correctly.
        # The actual column names are like 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'BBL_5_2.0' etc.
        # These are NOT lowercased by FeatureEngineer itself.
        self.assertIn('SMA_50', output_df.columns)
        self.assertIn('RSI_14', output_df.columns)
        self.assertIn('MACD_12_26_9', output_df.columns) # Example, check for actual MACD columns
        self.assertIn('BBL_5_2.0', output_df.columns)   # Example, check for actual BBands columns

        # Check for fundamental columns (as named by FeatureEngineer)
        self.assertIn('eps', output_df.columns)
        self.assertIn('pe_ratio', output_df.columns)
        self.assertEqual(output_df['eps'].iloc[0], 2.5)
        self.assertEqual(output_df['pe_ratio'].iloc[0], 20.0)

        # Check for other engineered features
        vol_window = self.dummy_config['fundamental_features']['volatility_window']
        mom_window = self.dummy_config['fundamental_features']['momentum_window']
        vol_trend_window = self.dummy_config['fundamental_features']['volume_trend_window']
        self.assertIn(f'volatility_{vol_window}d', output_df.columns)
        self.assertIn(f'momentum_{mom_window}d', output_df.columns)
        self.assertIn(f'volume_trend_{vol_trend_window}d', output_df.columns)

        # Check for OHLCV columns (should be TitleCased by FeatureEngineer at the end)
        self.assertIn('Open', output_df.columns)
        self.assertIn('Close', output_df.columns)

        # Check NaN handling - first few rows of rolling features might be NaN
        # but ffill/bfill in FeatureEngineer should handle them.
        # Check a column that's likely to have leading NaNs from TA calc before ffill/bfill
        # e.g. SMA_50 on 60 rows of data. First 49 will be NaN from SMA, then ffill/bfill.
        # The ffill().bfill() at the end of run_feature_engineering should eliminate all NaNs.
        # However, if a column is ALL NaN (e.g. SMA_200 on 60 rows), it will remain all NaN.
        # This test uses SMA_50 on 60 rows, so it should NOT be all NaN.
        self.assertFalse(output_df['SMA_50'].isnull().all(), "SMA_50 should not be all NaN after fill")
        # A more robust check would be that no column is all NaN, except if input data was too short for it.
        # For this test, with 60 rows, most indicators should compute.
        for col in output_df.columns:
            if output_df[col].isnull().all():
                 print(f"Warning: Column {col} is all NaN in output of test_full_processing_with_mocked_yf.")
        # A simple check for any NaNs after processing (should ideally be none)
        self.assertFalse(output_df.isnull().values.any(), "No NaNs should be present in the final output after ffill/bfill.")


    @patch('yfinance.Ticker')
    def test_yf_fetch_failure(self, MockTicker):
        # Simulate yf.Ticker().info raising an exception or returning empty/missing info
        mock_instance = MockTicker.return_value
        # Option 1: Raise an exception
        # mock_instance.info = MagicMock(side_effect=Exception("Simulated yf API error"))
        # Option 2: Return a dictionary with missing keys
        mock_instance.info = {'someOtherInfo': 'value'}


        self._create_dummy_csv("FAIL.csv")
        results = self.engineer.run_feature_engineering(self.input_dir, self.output_dir)
        self.assertEqual(len(results), 1)

        output_df = pd.read_csv(results[0])
        self.assertIn('eps', output_df.columns)
        self.assertIn('pe_ratio', output_df.columns)
        # Check if defaults were applied
        self.assertTrue((output_df['eps'] == self.dummy_config['default_eps']).all())
        self.assertTrue((output_df['pe_ratio'] == self.dummy_config['default_pe_ratio']).all())

    def test_output_file_creation(self):
        self._create_dummy_csv("TEST_OUT.csv")
        self.engineer.run_feature_engineering(self.input_dir, self.output_dir)
        expected_output_path = os.path.join(self.output_dir, "TEST_OUT.csv")
        self.assertTrue(os.path.exists(expected_output_path))

if __name__ == '__main__':
    unittest.main()
