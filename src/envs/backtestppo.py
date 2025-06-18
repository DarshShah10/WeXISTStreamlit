import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.trading_env import MultiStockTradingEnv
from src.utils import load_config # Import config loader
import datetime # For timestamped output directories

class RecurrentPPOBacktester:
    def __init__(self, model_path, env_path, output_dir=None, config=None, ticker_list=None):
        self.model_path = model_path
        self.env_path = env_path # Path to VecNormalize stats

        if config is None:
            self.config = load_config()
        else:
            self.config = config

        self.ticker_list = ticker_list if ticker_list is not None else self.config.get('default_ticker_list', [])
        if not self.ticker_list:
            raise ValueError("Ticker list must be provided either as an argument or in the config file.")

        if output_dir is None:
            project_root = self.config.get('PROJECT_ROOT', '.')
            log_dir = self.config.get('log_dir', 'logs')
            backtest_base_dir = self.config.get('backtest_output_dir', 'backtest_results')

            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # Create a unique directory for this backtest run
            self.output_dir = os.path.join(project_root, log_dir, backtest_base_dir, f"recurrent_ppo_{model_name}_{timestamp}")
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def run_backtest(self, backtest_csv_path):
        df_backtest = pd.read_csv(backtest_csv_path)
        if "Date" not in df_backtest.columns:
            raise ValueError("Backtest CSV data must contain a 'Date' column.")
        df_backtest["date_col"] = pd.to_datetime(df_backtest["Date"])
        df_backtest.set_index("date_col", inplace=True)

        num_stocks_env = len(self.ticker_list)

        # Environment parameters from config, with fallbacks
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

        env_metrics_path = os.path.join(self.output_dir, 'recurrent_ppo_env_metrics.csv')

        backtest_env = MultiStockTradingEnv(
            df=df_backtest,
            num_stocks=num_stocks_env,
            config=self.config, # Pass the full config
            initial_amount=initial_amount,
            buy_cost_pct=buy_cost_pct_list,
            sell_cost_pct=sell_cost_pct_list,
            hmax_per_stock=hmax_per_stock_list,
            reward_scaling=reward_scale, # This is MultiStockTradingEnv's internal scaling
            tech_indicator_list=tech_indicators,
            lookback_window=lookback,
            training=False, # Explicitly set to False
            metrics_save_path=env_metrics_path,
            seed=env_seed
        )

        model = RecurrentPPO.load(self.model_path)

        venv = DummyVecEnv([lambda: backtest_env])
        vec_normalize_backtest = VecNormalize.load(self.env_path, venv)

        obs = vec_normalize_backtest.reset()
        done = False
        episode_info = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = vec_normalize_backtest.step(action)
            episode_info.append(infos[0] if isinstance(infos, list) else infos)

        df = pd.DataFrame(episode_info)
        df['current_date'] = pd.to_datetime(df['current_date'])
        df.set_index('current_date', inplace=True)
        df.dropna(subset=['portfolio_value'], inplace=True)
        
        df['portfolio_return'] = df['portfolio_value'].pct_change().fillna(0)
        df['cumulative_return'] = df['portfolio_value'] / df['portfolio_value'][0] - 1
        df['peak_value'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['peak_value'] - df['portfolio_value']) / df['peak_value']
        sharpe_ratio = df['portfolio_return'].mean() / (df['portfolio_return'].std() + 1e-9) * np.sqrt(252)
        downside_returns = df['portfolio_return'].clip(upper=0)
        sortino_ratio = df['portfolio_return'].mean() / (downside_returns.std() + 1e-9) * np.sqrt(252)
        max_drawdown = df['drawdown'].max() * 100

        df.to_csv(os.path.join(self.output_dir, "backtest_metrics.csv"))
        self._plot_results(df)

        # Return summary
        return {
            "Final Portfolio Value": f"${df['portfolio_value'].iloc[-1]:,.2f}",
            "Total Trades": int(df['total_trades'].iloc[-1]),
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.4f}",
            "Sortino Ratio": f"{sortino_ratio:.4f}",
            "Results Saved To": self.output_dir
        }

    def _plot_results(self, df):
        plots = [
            (df['portfolio_value'], 'Portfolio Value', 'Portfolio Value ($)', 'darkblue'),
            (df['cumulative_return'] * 100, 'Cumulative Return', 'Return (%)', 'forestgreen'),
            (df['drawdown'] * 100, 'Drawdown', 'Drawdown (%)', 'firebrick')
        ]
        
        for data, title, ylabel, color in plots:
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.plot(df.index, data, label=title, color=color)
            plt.title(f'{title} Over Time', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.5, linestyle='dashed')
            plt.savefig(os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}_chart.png"))
            plt.close()
