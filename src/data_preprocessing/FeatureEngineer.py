# src/data_preprocessing/FeatureEngineer.py
import os
import pandas as pd
import numpy as np
import yfinance as yf # For fetching EPS/PE
import pandas_ta as ta # For technical indicators

class StockFeatureEngineer:
    def __init__(self, config=None):
        """
        Initializes the StockFeatureEngineer.
        :param config: Configuration dictionary. Expected to contain 'tech_indicator_list',
                       and potentially parameters for fundamental features or indicators.
        """
        self.config = config
        if not self.config:
            print("Warning: StockFeatureEngineer initialized without a config. Using default settings.")
            # Define a minimal default config for standalone testing
            self.config = {
                'tech_indicator_list': ['sma50', 'sma200', 'rsi', 'macd', 'bbands', 'obv', 'stoch'], # Example list
                'fundamental_features': { # Example structure for fundamental feature params
                    'volatility_window': 30,
                    'momentum_window': 10,
                    'volume_trend_window': 5
                },
                'PROJECT_ROOT': '.',
                'data_dir': 'data_test_feature_engineer',
                'processed_stock_data_dir': 'processed_stock_input', # Input for this test
                'feature_engineered_data_dir': 'feature_engineered_output' # Output for this test
            }

        # Standardize tech indicator list from config to match pandas_ta output patterns if needed
        # For now, assume config list is reasonably close or users will adjust config.
        # e.g. pandas_ta uses 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'OBV', 'STOCHk_14_3_3', 'STOCHd_14_3_3'
        # The config list like "sma50" needs to be mapped or pandas_ta strategy adjusted.
        # For simplicity, this implementation will try to call ta methods based on common names
        # and let pandas_ta create its default column names. These names must then be used by CombineDf and the env.

    def _get_ticker_from_filename(self, filename):
        """Extracts ticker from filename like 'AAPL.csv' or 'RELIANCE_NS.csv'"""
        base = os.path.splitext(filename)[0]
        # If we sanitized by replacing '.' with '_', reverse for yfinance if needed,
        # but yfinance often handles 'RELIANCE.NS' and 'RELIANCE_NS' similarly.
        # For now, assume base is the ticker yfinance can use.
        return base

    def run_feature_engineering(self, preprocessed_data_dir, feature_added_data_dir):
        """
        Loads preprocessed stock data, calculates technical indicators and other features,
        and saves the augmented data.

        :param preprocessed_data_dir: Directory containing preprocessed stock CSV files.
        :param feature_added_data_dir: Directory to save CSV files with added features.
        :return: List of successfully processed file paths.
        """
        if not os.path.exists(preprocessed_data_dir):
            print(f"Error: Input directory for feature engineering does not exist: {preprocessed_data_dir}")
            return []

        if not os.path.exists(feature_added_data_dir):
            os.makedirs(feature_added_data_dir, exist_ok=True)
            print(f"Created directory: {feature_added_data_dir}")

        processed_files_list = []
        input_csv_files = [f for f in os.listdir(preprocessed_data_dir) if f.endswith('.csv')]

        if not input_csv_files:
            print(f"No CSV files found in {preprocessed_data_dir}. Nothing to engineer features for.")
            return []

        print(f"[StockFeatureEngineer] Starting feature engineering for {len(input_csv_files)} files from {preprocessed_data_dir}.")
        print(f"Output directory: {feature_added_data_dir}")

        tech_indicator_config_list = self.config.get('tech_indicator_list', [])
        fundamental_params = self.config.get('fundamental_features', {})
        vol_window = fundamental_params.get('volatility_window', 30)
        mom_window = fundamental_params.get('momentum_window', 10)
        vol_trend_window = fundamental_params.get('volume_trend_window', 5)


        for csv_file in input_csv_files:
            input_filepath = os.path.join(preprocessed_data_dir, csv_file)
            output_filepath = os.path.join(feature_added_data_dir, csv_file)

            ticker_symbol = self._get_ticker_from_filename(csv_file)
            print(f"  Processing {ticker_symbol} (File: {csv_file})...")

            try:
                df = pd.read_csv(input_filepath)
                if df.empty:
                    print(f"    Warning: Preprocessed file {csv_file} is empty. Skipping.")
                    continue

                if 'Date' not in df.columns:
                    print(f"    Error: 'Date' column missing in {csv_file}. Skipping.")
                    continue
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                except Exception as e:
                    print(f"    Error processing Date column for {csv_file}: {e}. Skipping.")
                    continue

                # Ensure OHLCV columns are present (case-sensitive)
                ohlcv_map = {
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'
                }
                # If columns are lowercase from a source, map them. pandas_ta needs lowercase.
                df.rename(columns={v: k for k,v in ohlcv_map.items() if v in df.columns}, inplace=True)
                # Ensure they are lowercase for pandas_ta
                df.columns = [col.lower() for col in df.columns]


                # --- Calculate Technical Indicators using pandas_ta ---
                if tech_indicator_config_list:
                    print(f"    Calculating technical indicators: {tech_indicator_config_list}")
                    # Example: df.ta.strategy("common") would add many common indicators
                    # Or, add them individually based on config list:
                    for indicator_name in tech_indicator_config_list:
                        try:
                            if indicator_name == 'sma50': df.ta.sma(length=50, append=True)
                            elif indicator_name == 'sma200': df.ta.sma(length=200, append=True)
                            elif indicator_name == 'ema12': df.ta.ema(length=12, append=True)
                            elif indicator_name == 'ema26': df.ta.ema(length=26, append=True)
                            elif indicator_name == 'rsi': df.ta.rsi(append=True) # Default length 14
                            elif indicator_name == 'macd': df.ta.macd(append=True) # Appends MACD, MACDh, MACDs
                            elif indicator_name == 'bbands': df.ta.bbands(append=True) # Appends BBL, BBM, BBU, BBB, BBP
                            elif indicator_name == 'obv': df.ta.obv(append=True)
                            elif indicator_name == 'stoch': df.ta.stoch(append=True) # Appends STOCHk, STOCHd
                            elif indicator_name == 'cci': df.ta.cci(append=True)
                            elif indicator_name == 'adx': df.ta.adx(append=True) # Appends ADX, DMP, DMN
                            # Add more mappings as needed based on config names and pandas_ta methods/params
                            # e.g. 'stoch_k', 'stoch_d' are parts of 'stoch'
                            # 'bb_upper', 'bb_middle', 'bb_lower' are parts of 'bbands'
                            # 'volume_sma_20' -> df.ta.sma(close='volume', length=20, prefix='VOL', append=True) -> VOLSMA_20
                            else:
                                print(f"      Warning: Technical indicator '{indicator_name}' from config not explicitly mapped or recognized for pandas_ta.")
                        except Exception as e_ta:
                            print(f"      Error calculating TA '{indicator_name}' for {ticker_symbol}: {e_ta}")
                else:
                    print("    No technical indicators specified in config or config list is empty.")

                # --- Calculate Fundamental/Other Features ---
                print("    Calculating other features (volatility, momentum, etc.)...")
                # Volatility (annualized rolling std dev of returns)
                df['volatility'] = df['close'].pct_change().rolling(window=vol_window).std() * np.sqrt(252)
                df.rename(columns={'volatility': f'volatility_{vol_window}d'}, inplace=True)


                # Momentum (price difference over a period)
                df['momentum'] = df['close'].diff(periods=mom_window)
                df.rename(columns={'momentum': f'momentum_{mom_window}d'}, inplace=True)


                # Volume Trend (volume difference over a period)
                df['volume_trend'] = df['volume'].diff(periods=vol_trend_window)
                df.rename(columns={'volume_trend': f'volume_trend_{vol_trend_window}d'}, inplace=True)


                # EPS and P/E Ratio (fetched from yfinance - current values)
                # These will be single values broadcasted across all rows for the stock.
                # This is an approximation; historical daily fundamental data is complex to get.
                try:
                    ticker_obj = yf.Ticker(ticker_symbol) # Use original ticker symbol
                    info = ticker_obj.info
                    df['eps'] = info.get('trailingEps', np.nan)
                    df['pe_ratio'] = info.get('trailingPE', np.nan)
                    # Fill with a default if NaN, e.g. if stock is new or data unavailable
                    df['eps'].fillna(value=self.config.get('default_eps', 1.0), inplace=True)
                    df['pe_ratio'].fillna(value=self.config.get('default_pe_ratio', 15.0), inplace=True)

                except Exception as e_yf:
                    print(f"    Warning: Could not fetch yfinance info (EPS/PE) for {ticker_symbol}: {e_yf}")
                    df['eps'] = self.config.get('default_eps', 1.0) # Use default on error
                    df['pe_ratio'] = self.config.get('default_pe_ratio', 15.0)


                # --- Finalize DataFrame ---
                # Handle NaNs introduced by rolling windows or diffs (at the beginning of the series)
                # Columns created by pandas_ta usually handle their own NaNs internally or as per their calculation window.
                # However, our custom ones (volatility, momentum, eps, pe) might need it.
                df.ffill(inplace=True)
                df.bfill(inplace=True) # Fill any remaining NaNs at the very start

                df.reset_index(inplace=True) # Move 'Date' back to a column

                # Rename columns to ensure they match convention expected by trading_env.py
                # e.g., if pandas_ta created 'SMA_50', and env expects 'sma50_0' (if num_stocks > 1)
                # This renaming logic will be crucial and must align with how CombineDf.py names columns
                # and how MultiStockTradingEnv expects them (e.g. close_0, sma50_0, eps_0).
                # For now, this class saves features with generic names (e.g., 'SMA_50', 'eps').
                # The per-stock suffix (_0, _1) will be added by CombineDf.py.

                # Ensure original OHLCV columns are also present with consistent casing if changed for pandas_ta
                df.rename(columns={v.lower(): v for k,v in ohlcv_map.items()}, inplace=True) # Back to TitleCase for OHLCV

                df.to_csv(output_filepath, index=False)
                print(f"    Successfully engineered features for {ticker_symbol} and saved to {output_filepath}")
                processed_files_list.append(output_filepath)

            except pd.errors.EmptyDataError:
                print(f"    Warning: Preprocessed file {csv_file} was empty or became empty. Skipping.")
            except Exception as e:
                print(f"    Error engineering features for {csv_file} (Ticker: {ticker_symbol}): {e}")
                import traceback
                traceback.print_exc()


        print(f"[StockFeatureEngineer] Finished feature engineering. Successfully processed {len(processed_files_list)}/{len(input_csv_files)} files.")
        return processed_files_list

if __name__ == '__main__':
    print("Running StockFeatureEngineer example...")

    # Setup dummy config and directories for testing
    _PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    dummy_fe_config = {
        'PROJECT_ROOT': _PROJECT_ROOT_TEST,
        'data_dir': 'data_test_feature_engineer',
        'processed_stock_data_dir': 'processed_input_for_fe', # Input for this test
        'feature_engineered_data_dir': 'feature_engineered_output_fe', # Output for this test
        'tech_indicator_list': ['sma50', 'rsi', 'macd', 'bbands', 'obv'], # Simplified list for test
        'fundamental_features': {
            'volatility_window': 2, # Small window for test data
            'momentum_window': 1,
            'volume_trend_window': 1
        },
        'default_eps': 0.5, # Default EPS for testing if yf fetch fails for dummy ticker
        'default_pe_ratio': 10.0 # Default P/E for testing
    }

    test_fe_base_dir = os.path.join(dummy_fe_config['PROJECT_ROOT'], dummy_fe_config['data_dir'])
    test_fe_input_dir = os.path.join(test_fe_base_dir, dummy_fe_config['processed_stock_data_dir'])
    test_fe_output_dir = os.path.join(test_fe_base_dir, dummy_fe_config['feature_engineered_data_dir'])

    for dir_path in [test_fe_input_dir, test_fe_output_dir]:
        if os.path.exists(dir_path):
            for f_name in os.listdir(dir_path): os.remove(os.path.join(dir_path, f_name))
        else: os.makedirs(dir_path)

    # Create a dummy preprocessed CSV file (needs enough data for TA)
    dates = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 11)]) # 10 days of data
    dummy_ohlcv_data = {
        'Date': dates,
        'Open': np.random.uniform(100, 102, size=len(dates)),
        'High': np.random.uniform(102, 104, size=len(dates)),
        'Low': np.random.uniform(98, 100, size=len(dates)),
        'Close': np.random.uniform(100, 103, size=len(dates)),
        'Volume': np.random.randint(10000, 20000, size=len(dates))
    }
    # Make High > Open/Close and Low < Open/Close
    dummy_ohlcv_data['High'] = np.maximum(dummy_ohlcv_data['High'], dummy_ohlcv_data['Open'])
    dummy_ohlcv_data['High'] = np.maximum(dummy_ohlcv_data['High'], dummy_ohlcv_data['Close'])
    dummy_ohlcv_data['Low'] = np.minimum(dummy_ohlcv_data['Low'], dummy_ohlcv_data['Open'])
    dummy_ohlcv_data['Low'] = np.minimum(dummy_ohlcv_data['Low'], dummy_ohlcv_data['Close'])

    # Use a real ticker for yfinance info to work in test, e.g. 'MSFT'
    # If using a non-existent ticker, yf.Ticker(ticker).info will fail.
    # For the test, let's use a ticker that's likely to exist.
    test_ticker_symbol = 'MSFT'
    pd.DataFrame(dummy_ohlcv_data).to_csv(os.path.join(test_fe_input_dir, f'{test_ticker_symbol}.csv'), index=False)
    print(f"Created dummy preprocessed file: {test_ticker_symbol}.csv in {test_fe_input_dir}")

    engineer = StockFeatureEngineer(config=dummy_fe_config)

    print(f"Test input directory (preprocessed): {test_fe_input_dir}")
    print(f"Test output directory (features added): {test_fe_output_dir}")

    engineered_files = engineer.run_feature_engineering(test_fe_input_dir, test_fe_output_dir)
    print(f"Files successfully feature-engineered in test: {engineered_files}")

    for file_path in engineered_files:
        if os.path.exists(file_path):
            print(f"Verified: File '{file_path}' exists.")
            df_check = pd.read_csv(file_path)
            print(f"Data for {os.path.basename(file_path)} (first 3 rows):
{df_check.head(3)}")
            print(f"Columns in {os.path.basename(file_path)}: {df_check.columns.tolist()}")
            # Check for NaNs (some are expected at start due to lookback windows)
            # print(f"NaN sum per column:
{df_check.isnull().sum()}")
        else:
            print(f"Error: Feature-engineered file '{file_path}' NOT found.")

    print("StockFeatureEngineer example finished.")
