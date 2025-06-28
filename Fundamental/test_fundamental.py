import os
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from src.utils import load_config

class BacktestFundamental:
    def __init__(self, model_path=None, nifty_url=None, config=None):
        if config is None:
            self.config = load_config()
            print("BacktestFundamental: Loaded global config.")
        else:
            self.config = config
            print("BacktestFundamental: Using provided config.")

        # Determine nifty_url: argument > config > default error/placeholder
        if nifty_url:
            self.nifty_url = nifty_url
        else:
            self.nifty_url = self.config.get('nifty_url')

        if not self.nifty_url:
            raise ValueError("NIFTY URL must be provided either as an argument or in the config file ('nifty_url').")

        self.num_stocks_for_prediction = self.config.get('fundamental_num_stocks_to_predict_on', 500) # How many NIFTY stocks to run prediction on
        self.num_stocks_to_select = self.config.get('fundamental_num_stocks_to_select_final_backtest', 10) # How many top stocks to show

        if model_path:
            self.model_path_to_load = model_path
        else:
            project_root = self.config.get('PROJECT_ROOT', '.')
            model_dir = self.config.get('model_dir', 'models')
            xgb_filename = self.config.get('xgboost_model_filename', 'xgboost_fundamental_model.bin')
            self.model_path_to_load = os.path.join(project_root, model_dir, xgb_filename)

        print(f"BacktestFundamental: Model path to load: {self.model_path_to_load}")

        # Features list should ideally be consistent with the one used for training the loaded model.
        # For now, it's hardcoded, mirroring what was in StockPipeline.
        # Consider making this configurable or loading from model metadata if possible in future.
        self.features = [
            "marketCap", "trailingPE", "forwardPE", "pegRatio",
            "priceToBook", "trailingEps", "forwardEps",
            "returnOnEquity", "debtToEquity", "dividendYield",
            "beta", "revenueGrowth", "earningsGrowth"
        ]
        self.model = None
        
    def load_model(self):
        """Load the pre-trained XGBoost model"""
        try:
            model = xgb.XGBRegressor() # Initialize new model instance
            model.load_model(self.model_path_to_load)
            print(f"Successfully loaded model from {self.model_path_to_load}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path_to_load}")
            return None
        except Exception as e:
            print(f"Error loading XGBoost model from {self.model_path_to_load}: {e}")
            return None
        
    def fetch_nifty_list(self):
        """Fetch NIFTY list and process symbols"""
        print(f"Fetching NIFTY list from: {self.nifty_url}")
        try:
            if os.path.exists(self.nifty_url): # Check if it's a local file path
                df_nse = pd.read_csv(self.nifty_url)
            else: # Assume it's a URL
                df_nse = pd.read_csv(self.nifty_url)
        except Exception as e:
            print(f"Error fetching or reading NIFTY list from {self.nifty_url}: {e}")
            return pd.DataFrame(columns=["Yahoo_Ticker"])

        df_nse.rename(columns={"Symbol": "SYMBOL"}, inplace=True)
        if "SYMBOL" in df_nse.columns:
            df_nse["Yahoo_Ticker"] = df_nse["SYMBOL"].apply(lambda x: f"{str(x).strip()}.NS" if pd.notna(x) else None)
            df_nse.dropna(subset=["Yahoo_Ticker"], inplace=True)
        else:
            print("Warning: 'SYMBOL' column not found in NIFTY list. Cannot generate Yahoo Tickers.")
            df_nse["Yahoo_Ticker"] = None
        return df_nse
        
    def collect_current_metrics(self, tickers):
        """Collect current fundamental metrics for predictions"""
        current_features_list = []
        print(f"Collecting current metrics for {len(tickers)} tickers...")
        for ticker_count, ticker in enumerate(tickers):
            if ticker_count % 50 == 0 and ticker_count > 0: # Print progress
                print(f"  Processed {ticker_count}/{len(tickers)} tickers for metrics...")
            try:
                stock = yf.Ticker(ticker)
                info = stock.info # Fetch info once

                metrics = {"Ticker": ticker}
                for feature_key in self.features:
                    metrics[feature_key] = info.get(feature_key, np.nan)
                current_features_list.append(metrics)
            except Exception as e:
                print(f"  Error processing yfinance info for {ticker}: {e}")
                # Append with NaNs for this ticker to maintain structure if needed, or skip
                metrics = {"Ticker": ticker}
                for feature_key in self.features: metrics[feature_key] = np.nan
                current_features_list.append(metrics) # Add with NaNs

        df_current_features = pd.DataFrame(current_features_list)

        # Convert feature columns to numeric and handle NaNs by filling with 0 or median/mean
        # This is critical as model.predict cannot handle NaNs.
        for col in self.features:
            if col in df_current_features.columns:
                df_current_features[col] = pd.to_numeric(df_current_features[col], errors='coerce')
                # For simplicity, fill NaNs with 0. A more robust approach might use medians from a training set.
                df_current_features[col].fillna(0, inplace=True)
            else: # Should not happen if self.features is defined correctly
                print(f"Warning: Feature column '{col}' expected but not found after collection. Creating and filling with 0.")
                df_current_features[col] = 0

        return df_current_features

    def predict_scores(self, df): # Renamed from predict_returns for clarity
        """Generate prediction scores using the pre-trained model"""
        if self.model is None:
            print("Error: Model not loaded. Cannot generate predictions.")
            return df # Return original df or an empty one with PredictedScore

        if df.empty or not all(f in df.columns for f in self.features):
            print("Warning: DataFrame for prediction is empty or missing required feature columns. Cannot predict.")
            df["PredictedScore"] = 0.0 # Add dummy score column
            return df

        X_latest = df[self.features]
        df["PredictedScore"] = self.model.predict(X_latest)
        return df.sort_values("PredictedScore", ascending=False)

    def run_backtest(self):
        """Execute the backtesting (prediction) pipeline"""
        print("--- Running Fundamental Backtest/Prediction ---")
        self.model = self.load_model()
        if self.model is None:
            print("Aborting backtest due to model loading failure.")
            return pd.DataFrame() # Return empty DataFrame

        df_nse = self.fetch_nifty_list()
        if df_nse.empty or "Yahoo_Ticker" not in df_nse.columns or df_nse["Yahoo_Ticker"].isnull().all():
            print("Error: Could not fetch or process NIFTY ticker list for backtest. Aborting.")
            return pd.DataFrame()

        tickers_for_prediction = df_nse["Yahoo_Ticker"].tolist()[:self.num_stocks_for_prediction]
        
        current_data_df = self.collect_current_metrics(tickers_for_prediction)
        if current_data_df.empty:
            print("Error: No current metrics collected for tickers. Aborting.")
            return pd.DataFrame()

        predictions_df = self.predict_scores(current_data_df)
        
        top_stocks = predictions_df.head(self.num_stocks_to_select)
        print(f"--- Fundamental Backtest Finished ---")
        print(f"Top {self.num_stocks_to_select} stocks based on predicted score:")
        print(top_stocks[["Ticker", "PredictedScore"]])
        return top_stocks[["Ticker", "PredictedScore"]]

if __name__ == '__main__':
    print("Running test_fundamental.py standalone example...")

    dummy_nifty_data_backtest = {'SYMBOL': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN']}
    dummy_nifty_df_backtest = pd.DataFrame(dummy_nifty_data_backtest)
    dummy_nifty_csv_path_backtest = "dummy_nifty_list_fundamental_backtest.csv"
    dummy_nifty_df_backtest.to_csv(dummy_nifty_csv_path_backtest, index=False)
    print(f"Created dummy NIFTY list for backtest: {dummy_nifty_csv_path_backtest}")

    # This assumes Fundamental.py's __main__ was run and saved a model.
    # Path needs to match where that test model was saved.
    _PROJECT_ROOT_FOR_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    test_config_backtest = {
        'nifty_url': dummy_nifty_csv_path_backtest,
        'fundamental_num_stocks_to_predict_on': 5, # Predict on fewer for test
        'fundamental_num_stocks_to_select_final_backtest': 3,
        'PROJECT_ROOT': _PROJECT_ROOT_FOR_TEST,
        'model_dir': 'models_test_fundamental', # Should match StockPipeline's test output
        'xgboost_model_filename': 'test_xgb_fundamental_model.bin' # Should match StockPipeline's test output
    }
    print(f"Test config for BacktestFundamental: {test_config_backtest}")

    # Construct the expected model path based on the test config
    expected_model_path_for_test = os.path.join(
        test_config_backtest['PROJECT_ROOT'],
        test_config_backtest['model_dir'],
        test_config_backtest['xgboost_model_filename']
    )
    print(f"Expected model path for this test: {expected_model_path_for_test}")
    if not os.path.exists(expected_model_path_for_test):
        print(f"Warning: Test model file '{expected_model_path_for_test}' not found. "
              "Run Fundamental.py's __main__ block first to create it, or ensure path is correct.")
        # Create a dummy model file so xgb.load_model doesn't immediately fail
        # In a real test, this model should be compatible with the features.
        temp_xgb_model = xgb.XGBRegressor()
        # Create dummy data for fitting a minimal model
        dummy_X = pd.DataFrame(np.random.rand(10, len(BacktestFundamental(config=test_config_backtest).features)), columns=BacktestFundamental(config=test_config_backtest).features)
        dummy_y = pd.Series(np.random.rand(10))
        temp_xgb_model.fit(dummy_X, dummy_y)
        os.makedirs(os.path.dirname(expected_model_path_for_test), exist_ok=True)
        temp_xgb_model.save_model(expected_model_path_for_test)
        print(f"Created a temporary dummy XGBoost model at: {expected_model_path_for_test} for testing purposes.")


    try:
        # Test with config-based model path loading
        backtester = BacktestFundamental(config=test_config_backtest)
        top_stocks_output = backtester.run_backtest()
        
        if top_stocks_output is not None and not top_stocks_output.empty:
            print("
Standalone backtest run successful!")
            print("Top predicted stocks (using config for model path):")
            print(top_stocks_output)
        else:
            print("
Standalone backtest run completed, but no stocks were selected or an error occurred.")

        # Test with explicit model_path override
        # backtester_explicit_path = BacktestFundamental(
        #     model_path=expected_model_path_for_test, # Explicitly provide path
        #     config=test_config_backtest
        # )
        # top_stocks_explicit = backtester_explicit_path.run_backtest()
        # if top_stocks_explicit is not None and not top_stocks_explicit.empty:
        #     print("
Standalone backtest run successful with explicit model path!")
        #     print(top_stocks_explicit)

    except Exception as e:
        print(f"An error occurred during the standalone backtest run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_nifty_csv_path_backtest):
            os.remove(dummy_nifty_csv_path_backtest)
            print(f"Cleaned up dummy NIFTY list: {dummy_nifty_csv_path_backtest}")
        # Optionally, clean up the dummy model created only for this test if it was a generic placeholder
        # Be careful not to delete a model saved by Fundamental.py's actual test run if that's intended to be reused.
        # if "temp_xgb_model" in locals() and os.path.exists(expected_model_path_for_test): # A way to check if it was a temp model
        #     print(f"Note: A dummy model was created for this test run at {expected_model_path_for_test}. Consider cleaning it up if it's not the one from Fundamental.py's test.")


    print("test_fundamental.py standalone example finished.")
