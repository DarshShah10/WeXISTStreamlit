import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Environment.New_Trading_Env import MultiStockTradingEnv

class TD3Backtester:
    def __init__(self, model_path, env_path, output_dir):
        self.model_path = model_path
        self.env_path = env_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_metrics(self, df):
        # Basic metrics
        df['daily_returns'] = df['portfolio_value'].pct_change()
        df['cumulative_return'] = (df['portfolio_value'] / df['portfolio_value'].iloc[0]) - 1
        df['drawdown'] = (df['portfolio_value'].cummax() - df['portfolio_value']) / df['portfolio_value'].cummax()
        
        # Risk metrics
        annual_return = df['daily_returns'].mean() * 252
        annual_volatility = df['daily_returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        # Sortino Ratio
        downside_returns = df[df['daily_returns'] < 0]['daily_returns']
        downside_volatility = downside_returns.std() * np.sqrt(252)  # Annualized downside deviation
        sortino_ratio = annual_return / downside_volatility if downside_volatility != 0 else 0
        
        # Drawdown analysis
        max_drawdown = df['drawdown'].max() * 100
        avg_drawdown = df['drawdown'].mean() * 100
        
        # Trading metrics
        winning_trades = len(df[df['daily_returns'] > 0])
        losing_trades = len(df[df['daily_returns'] < 0])
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'Annual Return': f"{annual_return*100:.2f}%",
            'Annual Volatility': f"{annual_volatility*100:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Sortino Ratio': f"{sortino_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Average Drawdown': f"{avg_drawdown:.2f}%",
            'Win Rate': f"{win_rate*100:.2f}%",
            'Total Trades': winning_trades + losing_trades,
            'Winning Trades': winning_trades,
        }


    def create_plots(self, df):
        """Create various trading analysis plots"""
        plots = {}
        
        # Portfolio Value Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['portfolio_value'], color='blue', label='Portfolio Value')
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.grid(True)
        plt.tight_layout()
        plots['portfolio_value'] = fig

        # Returns Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['daily_returns'].dropna(), bins=50, ax=ax)
        ax.set_title('Distribution of Daily Returns')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plots['returns_dist'] = fig

        # Drawdown Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.3)
        ax.set_title('Portfolio Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        plt.tight_layout()
        plots['drawdown'] = fig

        # Trading Activity
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df.index, df['trades'], color='green', alpha=0.6)
        ax.set_title('Daily Trading Activity')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Trades')
        plt.tight_layout()
        plots['trading_activity'] = fig

        return plots

    def run_backtest(self, backtest_csv_path):
        # Load backtesting data
        df_backtest = pd.read_csv(backtest_csv_path)
        df_backtest["date"] = pd.to_datetime(df_backtest["Date"])
        df_backtest.set_index("date", inplace=True)

        # Create and run backtest environment
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

        # Load model and normalize environment
        model = TD3.load(self.model_path)
        venv = DummyVecEnv([lambda: backtest_env])
        vec_normalize_backtest = VecNormalize.load(self.env_path, venv)

        # Run simulation
        obs = vec_normalize_backtest.reset()
        done = False
        episode_info = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = vec_normalize_backtest.step(action)
            episode_info.append(infos[0] if isinstance(infos, list) else infos)

        # Process results
        df = pd.DataFrame(episode_info)
        df['current_date'] = pd.to_datetime(df['current_date'])
        df.set_index('current_date', inplace=True)
        
        # Calculate metrics and create plots
        metrics = self.calculate_metrics(df)
        plots = self.create_plots(df)
        
        # Save results
        df.to_csv(os.path.join(self.output_dir, "backtest_results.csv"))
        for name, fig in plots.items():
            fig.savefig(os.path.join(self.output_dir, f"{name}.png"))
            plt.close(fig)

        return {
            'metrics': metrics,
            'df': df,
            'plots_dir': self.output_dir
        }