# src/envs/base_backtester.py
import os
import abc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils import load_config # Assuming get_config is an alias or load_config is used
from src.envs.trading_env import MultiStockTradingEnv

class BaseBacktester(abc.ABC):
    def __init__(self, model_path: str, env_stats_path: str = None,
                 output_dir_suffix: str = None, config: dict = None, ticker_list: list[str] = None):
        """
        Initializes the BaseBacktester.

        :param model_path: Path to the trained RL model file.
        :param env_stats_path: Optional path to VecNormalize statistics.
        :param output_dir_suffix: Optional suffix to append to the output directory name.
        :param config: Optional configuration dictionary. If None, loads default.
        :param ticker_list: Optional list of tickers. If None, uses default from config.
        """
        self.model_path = model_path
        self.env_stats_path = env_stats_path

        if config is None:
            self.config = load_config()
        else:
            self.config = config

        self.ticker_list = ticker_list if ticker_list is not None else self.config.get('default_ticker_list', [])
        if not self.ticker_list:
            raise ValueError("Ticker list must be provided either as an argument or in the config file.")

        # Determine and create unique output directory
        project_root = self.config.get('PROJECT_ROOT', '.')
        log_dir_base = self.config.get('log_dir', 'logs')
        backtest_output_base = self.config.get('backtest_output_dir', 'backtest_results')

        model_name_stem = os.path.splitext(os.path.basename(self.model_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        dir_name = f"{self._get_model_type_name()}_{model_name_stem}_{timestamp}"
        if output_dir_suffix:
            dir_name += f"_{output_dir_suffix}"

        self.output_dir = os.path.join(project_root, log_dir_base, backtest_output_base, dir_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Backtest output directory: {self.output_dir}")

    @abc.abstractmethod
    def _load_model(self, env_for_model: DummyVecEnv):
        """
        Loads the RL model. To be implemented by subclasses.
        :param env_for_model: The environment instance (possibly wrapped) needed for model loading.
        :return: The loaded model.
        """
        pass

    @abc.abstractmethod
    def _get_model_type_name(self) -> str:
        """
        Returns a string representation of the model type (e.g., "PPO", "TD3").
        """
        pass

    def _create_environment(self, df_backtest: pd.DataFrame) -> DummyVecEnv:
        """
        Creates and wraps the trading environment for backtesting.
        :param df_backtest: DataFrame containing the backtesting data.
        :return: A DummyVecEnv instance, potentially normalized.
        """
        print("Creating backtesting environment...")
        num_stocks_env = len(self.ticker_list)

        initial_amount = self.config.get('initial_amount', 100000.0)

        buy_cost_conf = self.config.get('buy_cost_pct', 0.001)
        buy_cost_pct_list = [buy_cost_conf] * num_stocks_env if isinstance(buy_cost_conf, float) else buy_cost_conf
        if len(buy_cost_pct_list) != num_stocks_env:
             buy_cost_pct_list = [buy_cost_pct_list[0] if buy_cost_pct_list else 0.001] * num_stocks_env

        sell_cost_conf = self.config.get('sell_cost_pct', 0.001)
        sell_cost_pct_list = [sell_cost_conf] * num_stocks_env if isinstance(sell_cost_conf, float) else sell_cost_conf
        if len(sell_cost_pct_list) != num_stocks_env:
            sell_cost_pct_list = [sell_cost_pct_list[0] if sell_cost_pct_list else 0.001] * num_stocks_env

        hmax_conf = self.config.get('hmax_per_stock', 1000)
        hmax_per_stock_list = [hmax_conf] * num_stocks_env if isinstance(hmax_conf, (int, float)) else hmax_conf
        if len(hmax_per_stock_list) != num_stocks_env:
            hmax_per_stock_list = [hmax_per_stock_list[0] if hmax_per_stock_list else 1000] * num_stocks_env

        tech_indicators = self.config.get('tech_indicator_list', [])
        lookback = self.config.get('lookback_window', 30)
        reward_scale = self.config.get('reward_scaling', 1e-4) # General env reward scaling
        env_seed = self.config.get('random_seed', None)

        env_metrics_path = os.path.join(self.output_dir, f'{self._get_model_type_name()}_env_step_metrics.csv')

        raw_env = MultiStockTradingEnv(
            df=df_backtest,
            num_stocks=num_stocks_env,
            config=self.config,
            initial_amount=initial_amount,
            buy_cost_pct=buy_cost_pct_list,
            sell_cost_pct=sell_cost_pct_list,
            hmax_per_stock=hmax_per_stock_list,
            reward_scaling=reward_scale,
            tech_indicator_list=tech_indicators, # Passed to MultiStockTradingEnv constructor
            lookback_window=lookback,
            training=False,
            metrics_save_path=env_metrics_path,
            seed=env_seed
        )

        vec_env = DummyVecEnv([lambda: raw_env])
        if self.env_stats_path and os.path.exists(self.env_stats_path):
            print(f"Loading VecNormalize stats from: {self.env_stats_path}")
            vec_env = VecNormalize.load(self.env_stats_path, vec_env)
            vec_env.training = False  # Important: set to False for inference
            vec_env.norm_reward = False # Reward normalization not applied during inference
        else:
            print("Warning: No VecNormalize stats path provided or file not found. Using unnormalized environment.")

        return vec_env

    def run_full_backtest(self, backtest_data_source: (str | pd.DataFrame)) -> tuple[dict, pd.DataFrame]:
        """
        Orchestrates the entire backtesting process.
        :param backtest_data_source: Path to the backtest CSV data or a DataFrame.
        :return: Tuple containing (metrics_dict, results_dataframe).
        """
        print(f"\n--- Starting Full Backtest for Model: {self.model_path} ---")

        if isinstance(backtest_data_source, str):
            if not os.path.exists(backtest_data_source):
                raise FileNotFoundError(f"Backtest data CSV not found: {backtest_data_source}")
            print(f"Loading backtest data from CSV: {backtest_data_source}")
            df_backtest = pd.read_csv(backtest_data_source)
        elif isinstance(backtest_data_source, pd.DataFrame):
            df_backtest = backtest_data_source.copy()
            print("Using provided DataFrame for backtest data.")
        else:
            raise TypeError("backtest_data_source must be a file path (str) or pandas DataFrame.")

        if "Date" not in df_backtest.columns:
            raise ValueError("Backtest data must contain a 'Date' column.")
        try:
            df_backtest["Date"] = pd.to_datetime(df_backtest["Date"])
        except Exception as e:
            raise ValueError(f"Could not parse 'Date' column in backtest data: {e}")
        # No set_index here, MultiStockTradingEnv handles Date internally if needed from combinedf

        # 1. Create Environment
        vec_env = self._create_environment(df_backtest)

        # 2. Load Model
        print("Loading model...")
        model = self._load_model(vec_env) # Pass the created (possibly normalized) vec_env

        # 3. Run Simulation
        print("Running simulation...")
        obs = vec_env.reset()
        terminated = False
        truncated = False
        episode_info = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated_arr, infos = vec_env.step(action)
            terminated = terminated_arr[0] # Assuming DummyVecEnv
            # truncated is not explicitly handled by SB3 for DummyVecEnv in same way as terminated,
            # but MultiStockTradingEnv sets truncated=True if current_step >= max_steps
            if infos[0].get('TimeLimit.truncated', False) or infos[0].get('Truncated', False): # Check for SB3 or custom truncation
                 truncated = True

            episode_info.append(infos[0])

        print("Simulation finished.")
        vec_env.close()

        # 4. Process Results
        results_df = pd.DataFrame(episode_info)
        if 'current_date' not in results_df.columns:
            print("Warning: 'current_date' not found in simulation results. Plotting x-axis may be affected.")
            results_df['current_date'] = pd.to_datetime(df_backtest['Date'].iloc[:len(results_df)]) # Fallback
        else:
            results_df['current_date'] = pd.to_datetime(results_df['current_date'], errors='coerce')

        results_df.set_index('current_date', inplace=True)
        results_df.sort_index(inplace=True) # Ensure chronological order

        # Save raw simulation results
        raw_results_path = os.path.join(self.output_dir, f"{self._get_model_type_name()}_raw_simulation_results.csv")
        results_df.to_csv(raw_results_path)
        print(f"Raw simulation results saved to: {raw_results_path}")

        # 5. Calculate Metrics
        metrics = self._calculate_and_log_metrics(results_df)

        # 6. Generate Plots
        self._generate_and_save_plots(results_df)

        print(f"--- Backtest Finished. Results saved in: {self.output_dir} ---")
        return metrics, results_df

    def _calculate_and_log_metrics(self, df_results: pd.DataFrame) -> dict:
        print("Calculating performance metrics...")
        metrics = {}
        if 'portfolio_value' not in df_results.columns:
            print("Warning: 'portfolio_value' column not found in results. Cannot calculate metrics.")
            return metrics

        df = df_results.copy() # Avoid modifying original
        df['daily_returns'] = df['portfolio_value'].pct_change().fillna(0)

        initial_portfolio_value = df['portfolio_value'].iloc[0]
        final_portfolio_value = df['portfolio_value'].iloc[-1]

        metrics['Initial Portfolio Value'] = f"${initial_portfolio_value:,.2f}"
        metrics['Final Portfolio Value'] = f"${final_portfolio_value:,.2f}"
        total_return_pct = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
        metrics['Total Return'] = f"{total_return_pct:.2f}%"

        # Time calculations
        num_days = len(df)
        num_trading_days_per_year = self.config.get('trading_days_per_year', 252) # Default to 252
        total_years = num_days / num_trading_days_per_year

        if total_years > 0:
            annual_return = ( (final_portfolio_value / initial_portfolio_value) ** (1/total_years) - 1 ) * 100
            metrics['Annualized Return'] = f"{annual_return:.2f}%"
        else:
            metrics['Annualized Return'] = "N/A (duration < 1 year)"

        annual_volatility = df['daily_returns'].std() * np.sqrt(num_trading_days_per_year) * 100
        metrics['Annualized Volatility'] = f"{annual_volatility:.2f}%"

        risk_free_rate_annual = self.config.get('risk_free_rate_annual', 0.0) # Annual RFR
        sharpe_ratio = (df['daily_returns'].mean() * num_trading_days_per_year - risk_free_rate_annual) / (df['daily_returns'].std() * np.sqrt(num_trading_days_per_year)) if df['daily_returns'].std() != 0 else 0
        metrics['Sharpe Ratio'] = f"{sharpe_ratio:.2f}"

        downside_returns = df[df['daily_returns'] < 0]['daily_returns']
        downside_deviation = downside_returns.std() * np.sqrt(num_trading_days_per_year)
        sortino_ratio = (df['daily_returns'].mean() * num_trading_days_per_year - risk_free_rate_annual) / downside_deviation if downside_deviation != 0 else 0
        metrics['Sortino Ratio'] = f"{sortino_ratio:.2f}"

        df['cumulative_max'] = df['portfolio_value'].cummax()
        df['drawdown_val'] = df['cumulative_max'] - df['portfolio_value']
        df['drawdown_pct'] = (df['drawdown_val'] / df['cumulative_max']).fillna(0) * 100 # Percentage
        metrics['Max Drawdown'] = f"{df['drawdown_pct'].max():.2f}%"

        if 'trades' in df.columns and df['trades'].sum() > 0 : # Assuming 'trades' column logs number of trades per step
            metrics['Total Trades'] = int(df['trades'].sum())
            # Further trade analysis would require more detailed trade log data (buy/sell price, quantity)
        else:
             metrics['Total Trades'] = "N/A (no 'trades' column in results_df or no trades made)"


        metrics_str = "\n--- Performance Metrics ---\n"
        for key, value in metrics.items():
            metrics_str += f"{key}: {value}\n"
        print(metrics_str)

        with open(os.path.join(self.output_dir, "performance_metrics.txt"), "w") as f:
            f.write(metrics_str)
        print("Performance metrics saved to performance_metrics.txt")
        return metrics

    def _generate_and_save_plots(self, df_results: pd.DataFrame):
        print("Generating and saving plots...")
        df = df_results.copy()

        # Plot 1: Portfolio Value Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['portfolio_value'], label='Portfolio Value', color='blue')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date'); plt.ylabel('Portfolio Value ($)')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "portfolio_value.png"))
        plt.close()

        # Plot 2: Drawdown Percentage
        if 'drawdown_pct' in df.columns: # Calculated in _calculate_and_log_metrics
            plt.figure(figsize=(12, 6))
            plt.fill_between(df.index, -df['drawdown_pct'], 0, label='Drawdown %', color='red', alpha=0.3)
            plt.title('Portfolio Drawdown Percentage Over Time')
            plt.xlabel('Date'); plt.ylabel('Drawdown (%)')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45); plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "drawdown_percentage.png"))
            plt.close()

        # Plot 3: Daily Returns Distribution
        if 'daily_returns' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['daily_returns'].dropna() * 100, bins=50, kde=True)
            plt.title('Distribution of Daily Returns')
            plt.xlabel('Daily Return (%)'); plt.ylabel('Frequency')
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "daily_returns_distribution.png"))
            plt.close()

        print("Plots generated and saved.")

# Example usage (for testing, would be in a separate test file or if __name__ == "__main__":)
# class MyPPOBacktester(BaseBacktester):
#     def _load_model(self, env_for_model):
#         from stable_baselines3 import PPO
#         print(f"Loading PPO model from {self.model_path}")
#         return PPO.load(self.model_path, env=env_for_model)
#     def _get_model_type_name(self) -> str:
#         return "PPO"

# if __name__ == '__main__':
#     # This requires a trained PPO model and its VecNormalize stats (if used)
#     # And a CSV file for backtesting data.
#     # Create dummy files/config for a full test.
#     print("BaseBacktester example (requires concrete implementation and test files).")
#     # dummy_config = load_config() # Load your actual project config
#     # backtester = MyPPOBacktester(
#     #     model_path="path/to/your/ppo_model.zip",
#     #     env_stats_path="path/to/your/vecnormalize.pkl", # Optional
#     #     config=dummy_config,
#     #     ticker_list=["AAPL", "MSFT"] # Example
#     # )
#     # backtester.run_full_backtest("path/to/your/backtest_data.csv")
