import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from Environment.New_Trading_Env import MultiStockTradingEnv
from New_Trading_Env import MultiStockTradingEnv

class RecurrentPPOBacktester:
    def __init__(self, model_path, env_path, output_dir):
        self.model_path = model_path
        self.env_path = env_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_backtest(self, backtest_csv_path):
        df_backtest = pd.read_csv(backtest_csv_path)
        df_backtest["date"] = pd.to_datetime(df_backtest["Date"])
        df_backtest.set_index("date", inplace=True)

        backtest_env = MultiStockTradingEnv(
            df=df_backtest,
            num_stocks=5,
            initial_amount=100000.00,
            buy_cost_pct=[0.001] * 5,
            sell_cost_pct=[0.001] * 5,
            hmax_per_stock=[1000] * 5,
            reward_scaling=1e-4,
            tech_indicator_list=[
                "sma50", "sma200", "ema12", "ema26", "macd", "rsi", "cci", "adx",
                "sok", "sod", "du", "dl", "vm", "bb_upper", "bb_lower", "bb_middle", "obv"
            ],
            lookback_window=30,
            training=False,
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
