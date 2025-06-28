import os
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split # Changed from manual split
from sklearn.metrics import mean_squared_error
from src.utils import load_config # For loading main config

class StockPipeline:
    def __init__(self, config=None, nifty_url=None): # nifty_url can be passed for testing or specific override
        if config is None:
            self.config = load_config()
            print("StockPipeline: Loaded global config.")
        else:
            self.config = config
            print("StockPipeline: Using provided config.")

        # Determine nifty_url: argument > config > default error/placeholder
        if nifty_url:
            self.nifty_url = nifty_url
        else:
            self.nifty_url = self.config.get('nifty_url')

        if not self.nifty_url:
            raise ValueError("NIFTY URL must be provided either as an argument or in the config file ('nifty_url').")

        self.num_stocks_to_process_for_training = self.config.get('fundamental_num_stocks_to_process', 50) # Reduced default for faster processing if not in config
        self.num_stocks_to_select_final = self.config.get('fundamental_num_stocks_to_select', 10)

        # Define model save path using config
        project_root = self.config.get('PROJECT_ROOT', '.')
        model_dir = self.config.get('model_dir', 'models') # Default 'models' if not in config
        xgb_filename = self.config.get('xgboost_model_filename', 'xgboost_fundamental_model.bin') # Default filename
        self.model_save_path = os.path.join(project_root, model_dir, xgb_filename)

        # Ensure the directory for saving the model exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        print(f"StockPipeline: Model will be saved to: {self.model_save_path}")

        # XGBoost hyperparameters from config, with defaults
        default_xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
        self.xgb_params = self.config.get('xgboost_fundamental_params', default_xgb_params)
        print(f"StockPipeline: XGBoost params: {self.xgb_params}")

        self.features = [ # These are keys expected from yf.Ticker().info
            "marketCap", "trailingPE", "forwardPE", "pegRatio",
            "priceToBook", "trailingEps", "forwardEps",
            "returnOnEquity", "debtToEquity", "dividendYield",
            "beta", "revenueGrowth", "earningsGrowth" # earningsQuarterlyGrowth renamed
        ] # Using yfinance info keys directly
        self.target_column_name = "dummy_target_1y_return" # Simplified target
        self.predictions = None
        self.model = None
        self.fundamental_data_df = None


    def fetch_nifty_list(self):
        """Step 1: Fetch NIFTY list and process symbols"""
        print(f"Fetching NIFTY list from: {self.nifty_url}")
        try:
            # Check if nifty_url is a local file path for testing
            if os.path.exists(self.nifty_url):
                df_nse = pd.read_csv(self.nifty_url)
            else: # Assume it's a URL
                df_nse = pd.read_csv(self.nifty_url)
        except Exception as e:
            print(f"Error fetching or reading NIFTY list from {self.nifty_url}: {e}")
            return pd.DataFrame(columns=["Yahoo_Ticker"]) # Return empty df with expected column

        df_nse.rename(columns={"Symbol": "SYMBOL"}, inplace=True) # Ensure consistency if "Symbol" is used
        # Ensure SYMBOL column exists before trying to apply lambda
        if "SYMBOL" in df_nse.columns:
            df_nse["Yahoo_Ticker"] = df_nse["SYMBOL"].apply(lambda x: f"{str(x).strip()}.NS" if pd.notna(x) else None)
            df_nse.dropna(subset=["Yahoo_Ticker"], inplace=True) # Remove rows where ticker could not be formed
        else:
            print("Warning: 'SYMBOL' column not found in NIFTY list. Cannot generate Yahoo Tickers.")
            df_nse["Yahoo_Ticker"] = None # Ensure column exists
        return df_nse


    def collect_current_fundamental_snapshot(self, tickers):
        """Step 2: Collect current fundamental metrics for each ticker using yf.Ticker().info"""
        all_data = []
        print(f"Collecting current fundamental snapshot for {len(tickers)} tickers...")
        for ticker_count, ticker in enumerate(tickers):
            if ticker_count % 20 == 0: # Print progress every 20 tickers
                print(f"  Processed {ticker_count}/{len(tickers)} tickers...")
            try:
                stock_info = yf.Ticker(ticker).info

                # Extract features defined in self.features
                # Use .get(key, np.nan) for robustness if a key is missing from info
                metrics = {"Ticker": ticker}
                for feature_key in self.features:
                    metrics[feature_key] = stock_info.get(feature_key, np.nan)

                # Add a dummy target variable
                metrics[self.target_column_name] = np.random.rand() # Random target between 0 and 1
                
                all_data.append(metrics)
            except Exception as e:
                print(f"  Error processing yfinance info for {ticker}: {e}")
        print(f"  Finished collecting fundamental snapshots.")
        return pd.DataFrame(all_data)

    def prepare_split_data(self, df):
        """Step 3: Prepare and split data into train/test sets"""
        df_clean = df.copy()
        # Ensure target column exists
        if self.target_column_name not in df_clean.columns:
             raise ValueError(f"Target column '{self.target_column_name}' not found in DataFrame.")

        # Convert all feature columns to numeric, coercing errors. This makes non-numeric into NaN.
        for col in self.features:
            if col in df_clean.columns:
                 df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            else: # If a feature column is missing, create it with NaNs
                print(f"Warning: Feature column '{col}' missing. Will be filled with NaNs.")
                df_clean[col] = np.nan

        df_clean = df_clean.dropna(subset=self.features + [self.target_column_name]) # Drop rows with NaN in features or target
        
        if df_clean.empty:
            raise ValueError("DataFrame is empty after NaN removal. Cannot train model.")

        X = df_clean[self.features]
        y = df_clean[self.target_column_name]
        
        # Using sklearn's train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.config.get('random_seed', 42))
        print(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        """Step 4: Train XGBoost model using parameters from config"""
        print(f"Training XGBoost model with params: {self.xgb_params}")
        model = xgb.XGBRegressor(**self.xgb_params) # Use params from config
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Step 5: Evaluate model performance"""
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE on test data: {rmse:.4f}")
        # Further evaluation metrics can be added here
        return rmse

    # collect_latest_metrics is similar to collect_current_fundamental_snapshot but without target
    # For this simplified version, we can reuse collect_current_fundamental_snapshot for prediction input
    # if the target column is just ignored by model.predict().
    # Or, create a variant that doesn't add the dummy target.
    def collect_data_for_prediction(self, tickers):
        """Step 6: Collect current data for making predictions (no target column needed)"""
        all_data = []
        print(f"Collecting current data for prediction for {len(tickers)} tickers...")
        for ticker in tickers:
            try:
                stock_info = yf.Ticker(ticker).info
                metrics = {"Ticker": ticker}
                for feature_key in self.features:
                    metrics[feature_key] = stock_info.get(feature_key, np.nan)
                all_data.append(metrics)
            except Exception as e:
                print(f"  Error processing yfinance info for {ticker} (prediction data): {e}")

        df_pred_data = pd.DataFrame(all_data)
        # Convert feature columns to numeric and handle NaNs by filling with mean/median or a placeholder
        # This is important because model.predict() cannot handle NaNs.
        for col in self.features:
            if col in df_pred_data.columns:
                df_pred_data[col] = pd.to_numeric(df_pred_data[col], errors='coerce')
                # Fill NaNs here, e.g. with median from training data if available, or 0 / mean
                # For simplicity in this example, let's fill with 0, but this should be more robust.
                df_pred_data[col].fillna(0, inplace=True)
            else:
                print(f"Warning: Feature column '{col}' missing for prediction. Filling with 0.")
                df_pred_data[col] = 0
        return df_pred_data


    def predict_scores(self, model, df_for_prediction):
        """Step 7: Generate scores (predictions) using the trained model"""
        if df_for_prediction.empty or not all(f in df_for_prediction.columns for f in self.features) :
            print("Warning: DataFrame for prediction is empty or missing feature columns. Cannot predict.")
            # Add a dummy PredictedReturn column if df_for_prediction is not empty but features are bad
            if not df_for_prediction.empty:
                 df_for_prediction["PredictedScore"] = 0.0
            return df_for_prediction

        # Ensure only features used for training are present and in correct order
        X_latest = df_for_prediction[self.features]
        df_for_prediction["PredictedScore"] = model.predict(X_latest)
        return df_for_prediction

    def run_pipeline(self):
        print("--- Running Fundamental Analysis Pipeline ---")
        df_nse = self.fetch_nifty_list()
        if df_nse.empty or "Yahoo_Ticker" not in df_nse.columns or df_nse["Yahoo_Ticker"].isnull().all():
            print("Error: Could not fetch or process NIFTY ticker list. Aborting pipeline.")
            return pd.DataFrame() # Return empty DataFrame

        tickers_for_training = df_nse["Yahoo_Ticker"].tolist()[:self.num_stocks_to_process_for_training]
        
        print(f"Number of tickers for training data collection: {len(tickers_for_training)}")
        self.fundamental_data_df = self.collect_current_fundamental_snapshot(tickers_for_training)
        
        if self.fundamental_data_df.empty:
            print("Error: No fundamental data collected. Aborting pipeline.")
            return pd.DataFrame()

        X_train, y_train, X_test, y_test = self.prepare_split_data(self.fundamental_data_df)
        
        self.model = self.train_model(X_train, y_train)
        print(f"Model trained. Saving model to: {self.model_save_path}")
        self.model.save_model(self.model_save_path) # Save the trained model
        
        print("Evaluating model...")
        self.evaluate_model(self.model, X_test, y_test)
        
        # For prediction, typically we'd use all tickers from NIFTY list or a predefined list
        all_available_tickers = df_nse["Yahoo_Ticker"].tolist()
        print(f"Collecting latest data for prediction on {len(all_available_tickers)} tickers...")
        latest_metrics_df = self.collect_data_for_prediction(all_available_tickers)
        
        if latest_metrics_df.empty:
            print("Error: No data collected for final predictions. Aborting.")
            return pd.DataFrame()

        print("Generating scores for all tickers...")
        self.predictions = self.predict_scores(self.model, latest_metrics_df)
        
        # Select top N stocks based on PredictedScore
        top_n_stocks = self.predictions.sort_values(
            by="PredictedScore", ascending=False
        ).head(self.num_stocks_to_select_final)
        
        print(f"--- Fundamental Analysis Pipeline Finished ---")
        print(f"Top {self.num_stocks_to_select_final} stocks based on predicted score:")
        print(top_n_stocks[["Ticker", "PredictedScore"]])
        return top_n_stocks[["Ticker", "PredictedScore"]]


if __name__ == '__main__':
    print("Running Fundamental.py standalone example...")

    # Create a dummy NIFTY list CSV for local testing
    dummy_nifty_data = {'SYMBOL': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'BAJFINANCE', 'BHARTIARTL', 'HINDUNILVR']}
    dummy_nifty_df = pd.DataFrame(dummy_nifty_data)
    dummy_nifty_csv_path = "dummy_nifty_list_fundamental_test.csv"
    dummy_nifty_df.to_csv(dummy_nifty_csv_path, index=False)
    print(f"Created dummy NIFTY list for test: {dummy_nifty_csv_path}")

    # Create a dummy config for this test
    # Note: PROJECT_ROOT is derived to be the actual project root assuming this script is in Fundamental/
    # This ensures that model_dir is created inside the project structure correctly.
    _PROJECT_ROOT_FOR_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    test_config = {
        'nifty_url': dummy_nifty_csv_path, # Use local dummy CSV
        'fundamental_num_stocks_to_process': 5, # Process fewer stocks for quick test
        'fundamental_num_stocks_to_select': 2,  # Select top 2
        'PROJECT_ROOT': _PROJECT_ROOT_FOR_TEST,
        'model_dir': 'models_test_fundamental', # Test-specific model directory
        'xgboost_model_filename': 'test_xgb_fundamental_model.bin',
        'xgboost_fundamental_params': { # Example of overriding XGBoost params
            'objective': 'reg:squarederror',
            'n_estimators': 20, # Fewer estimators for faster test
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
        'random_seed': 42 # For train_test_split consistency
    }
    print(f"Test config being used: {test_config}")

    # Ensure test model directory exists and is clean
    test_model_full_dir = os.path.join(test_config['PROJECT_ROOT'], test_config['model_dir'])
    if not os.path.exists(test_model_full_dir):
        os.makedirs(test_model_full_dir)
    else: # Clean up model file from previous test if it exists
        _test_model_file = os.path.join(test_model_full_dir, test_config['xgboost_model_filename'])
        if os.path.exists(_test_model_file):
            os.remove(_test_model_file)

    print(f"Test model directory: {test_model_full_dir}")


    try:
        pipeline = StockPipeline(config=test_config) # Pass the test config
        top_stocks = pipeline.run_pipeline()

        if top_stocks is not None and not top_stocks.empty:
            print("
Standalone pipeline run successful!")
            print("Top predicted stocks:")
            print(top_stocks)

            # Verify model was saved
            expected_model_path = pipeline.model_save_path
            if os.path.exists(expected_model_path):
                print(f"Model successfully saved to: {expected_model_path}")
            else:
                print(f"Error: Model file NOT found at: {expected_model_path}")
        else:
            print("
Standalone pipeline run completed, but no stocks were selected or an error occurred.")

    except Exception as e:
        print(f"An error occurred during the standalone pipeline run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy NIFTY CSV
        if os.path.exists(dummy_nifty_csv_path):
            os.remove(dummy_nifty_csv_path)
            print(f"Cleaned up dummy NIFTY list: {dummy_nifty_csv_path}")

    print("Fundamental.py standalone example finished.")
