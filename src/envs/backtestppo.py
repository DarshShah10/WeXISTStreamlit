import os
# import numpy as np # No longer directly used here
import pandas as pd # Still needed for type hinting
# import matplotlib.pyplot as plt # Handled by BaseBacktester
# import matplotlib.dates as mdates # Handled by BaseBacktester
from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # Handled by BaseBacktester
# from src.envs.trading_env import MultiStockTradingEnv # Handled by BaseBacktester
# from src.utils import load_config # Handled by BaseBacktester
# import datetime # Handled by BaseBacktester

from .base_backtester import BaseBacktester

class RecurrentPPOBacktester(BaseBacktester):
    def __init__(self, model_path: str, env_stats_path: str = None,
                 output_dir_suffix: str = None, config: dict = None, ticker_list: list[str] = None):
        super().__init__(
            model_path=model_path,
            env_stats_path=env_stats_path,
            output_dir_suffix=output_dir_suffix,
            config=config,
            ticker_list=ticker_list
        )

    def _load_model(self, env_for_model):
        """Loads the RecurrentPPO model."""
        print(f"Loading RecurrentPPO model from: {self.model_path}")
        return RecurrentPPO.load(self.model_path, env=env_for_model)

    def _get_model_type_name(self) -> str:
        return "RecurrentPPO"

    def run_backtest(self, backtest_data_source: (str | pd.DataFrame)) -> tuple[dict, pd.DataFrame]:
        """
        Runs the full backtest using the BaseBacktester's orchestration.
        
        :param backtest_data_source: Path to the backtest CSV data or a pandas DataFrame.
        :return: Tuple containing (metrics_dict, results_dataframe).
        """
        # The main logic is now in BaseBacktester.run_full_backtest
        # This method can include any PPO-specific pre- or post-backtest steps if needed in the future.
        return self.run_full_backtest(backtest_data_source=backtest_data_source)

# Example usage (can be kept for testing this specific backtester)
if __name__ == '__main__':
    print("Running RecurrentPPOBacktester example...")
    # This requires a trained RecurrentPPO model, its VecNormalize stats (if used),
    # a CSV file for backtesting data, and shared_config.yaml correctly set up.

    # --- Configuration for the test (similar to TD3Backtester's example) ---
    _DUMMY_MODEL_DIR = "models_test/rppo_test_models"
    os.makedirs(_DUMMY_MODEL_DIR, exist_ok=True)
    _DUMMY_MODEL_PATH = os.path.join(_DUMMY_MODEL_DIR, "dummy_rppo_model.zip")
    if not os.path.exists(_DUMMY_MODEL_PATH):
        with open(_DUMMY_MODEL_PATH, "w") as f:
            f.write("This is a dummy model file for RecurrentPPOBacktester test.")

    _DUMMY_ENV_STATS_PATH = None

    _DUMMY_DATA_DIR = "data_test/rppo_backtester_data"
    os.makedirs(_DUMMY_DATA_DIR, exist_ok=True)
    _DUMMY_BACKTEST_CSV = os.path.join(_DUMMY_DATA_DIR, "dummy_rppo_backtest_data.csv")

    try:
        # This conceptual example relies on proper setup of shared_config.yaml
        # and a comprehensive dummy_rppo_backtest_data.csv.
        print("Conceptual example: To run this effectively, provide a valid model, env_stats (opt),")
        print("and a backtest_data_source CSV that matches your environment's expected feature set.")
        print(f"Dummy model path: {_DUMMY_MODEL_PATH}")
        print(f"Dummy backtest data path: {_DUMMY_BACKTEST_CSV} (needs to be created with full features)")

        # Example instantiation (will likely fail at data validation if CSV is not comprehensive)
        # backtester = RecurrentPPOBacktester(
        #     model_path=_DUMMY_MODEL_PATH,
        #     env_stats_path=_DUMMY_ENV_STATS_PATH
        # )
        # if os.path.exists(_DUMMY_BACKTEST_CSV):
        #     metrics, results = backtester.run_backtest(backtest_data_source=_DUMMY_BACKTEST_CSV)
        #     print("
Backtest Metrics (RecurrentPPO):")
        #     for k, v in metrics.items():
        #         print(f"  {k}: {v}")
        # else:
        #      print(f"Skipping run_backtest as dummy data file '{_DUMMY_BACKTEST_CSV}' not found or not fully prepared.")

    except Exception as e:
        print(f"Error in RecurrentPPOBacktester example: {e}")
        import traceback
        traceback.print_exc()
        
    print("RecurrentPPOBacktester example finished.")
