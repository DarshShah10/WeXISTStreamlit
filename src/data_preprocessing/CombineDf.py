import os
import pandas as pd

class DataMerger:
    def __init__(self, post_processed_stock_dir, processed_macro_path, output_path_for_merger):
        self.post_processed_stock_dir = post_processed_stock_dir
        self.processed_macro_path = processed_macro_path
        self.output_path = output_path_for_merger # This is what DPP passes now

    def run_combine_data(self):
        print(f"[Dummy CombineDf] Called. Outputting to: {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Create a minimal CSV that MultiStockTradingEnv can load.
        # It needs Date, open_X, high_X, low_X, close_X, volume_X for each stock,
        # tech indicators, and macro columns.

        # Tech indicators list from train.py (used by MultiStockTradingEnv)
        TECH_INDICATOR_LIST = [
            "sma50", "sma200", "ema12", "ema26", "macd", "rsi", "cci", "adx",
            "sok", "sod", "du", "dl", "vm", "bb_upper", "bb_lower", "bb_middle", "obv"
        ]
        # Macro columns expected by MultiStockTradingEnv's _validate_data method
        MACRO_COLUMNS = ["snp500", "gold_price", "interest_rate"]

        # Determine number of stocks.
        # This is tricky. The actual pipeline processes files from post_processed_stock_dir.
        # The train.py script passes a list of tickers (DEFAULT_TICKER_LIST has 5).
        # For this dummy, we can either hardcode or try to infer.
        # The MultiStockTradingEnv receives num_stocks as an argument.
        # Let's use a fixed number of stocks for the dummy data generation,
        # e.g., 5, matching DEFAULT_TICKER_LIST length.
        num_dummy_stocks = 5

        # Create a DataFrame with a few days of data.
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'])
        data = {'Date': dates}

        # Add macro data
        for col in MACRO_COLUMNS:
            data[col] = [100 + i for i in range(len(dates))] # Simple ascending values

        # Add stock data (OHLCV and tech indicators)
        for i in range(num_dummy_stocks):
            data[f'open_{i}'] = [100 + i + j for j in range(len(dates))]
            data[f'high_{i}'] = [105 + i + j for j in range(len(dates))]
            data[f'low_{i}'] = [98 + i + j for j in range(len(dates))]
            data[f'close_{i}'] = [102 + i + j for j in range(len(dates))]
            data[f'volume_{i}'] = [1000 + i * 100 + j * 10 for j in range(len(dates))]
            # Adding other fundamental features expected by MultiStockTradingEnv's _validate_data
            data[f'eps_{i}'] = [1.0 + i*0.1] * len(dates)
            data[f'pe_ratio_{i}'] = [10.0 + i] * len(dates)
            data[f'volatility_30d_{i}'] = [0.02 + i*0.001] * len(dates)

            for tech_indicator in TECH_INDICATOR_LIST:
                data[f'{tech_indicator}_{i}'] = [50 + i + j for j in range(len(dates))]

        df_combined = pd.DataFrame(data)
        df_combined.to_csv(self.output_path, index=False)
        print(f"[Dummy CombineDf] Created dummy combined file: {self.output_path} with {num_dummy_stocks} dummy stocks and {len(dates)} days.")
        return self.output_path
