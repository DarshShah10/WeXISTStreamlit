import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.trading_env import MultiStockTradingEnv
from src.utils import load_config # Import config loader
import datetime # For timestamped output directories

class TD3Backtester:
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
            self.output_dir = os.path.join(project_root, log_dir, backtest_base_dir, f"td3_{model_name}_{timestamp}")
        else:
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
        # Ensure 'Date' column exists and is used for datetime conversion
        if "Date" not in df_backtest.columns:
            raise ValueError("Backtest CSV data must contain a 'Date' column.")
        df_backtest["date_col"] = pd.to_datetime(df_backtest["Date"]) # Use a temporary column name
        df_backtest.set_index("date_col", inplace=True) # Set index using the new column

        num_stocks_env = len(self.ticker_list)

        # Environment parameters from config, with fallbacks
        initial_amount = self.config.get('initial_amount', 100000.0)

        buy_cost_conf = self.config.get('buy_cost_pct', 0.001)
        buy_cost_pct_list = [buy_cost_conf] * num_stocks_env if isinstance(buy_cost_conf, float) else buy_cost_conf
        if len(buy_cost_pct_list) != num_stocks_env: # Fallback if list size mismatch
            buy_cost_pct_list = [buy_cost_pct_list[0] if buy_cost_pct_list else 0.001] * num_stocks_env

        sell_cost_conf = self.config.get('sell_cost_pct', 0.001)
        sell_cost_pct_list = [sell_cost_conf] * num_stocks_env if isinstance(sell_cost_conf, float) else sell_cost_conf
        if len(sell_cost_pct_list) != num_stocks_env: # Fallback
            sell_cost_pct_list = [sell_cost_pct_list[0] if sell_cost_pct_list else 0.001] * num_stocks_env

        hmax_conf = self.config.get('hmax_per_stock', 1000)
        hmax_per_stock_list = [hmax_conf] * num_stocks_env if isinstance(hmax_conf, (int, float)) else hmax_conf
        if len(hmax_per_stock_list) != num_stocks_env: # Fallback
            hmax_per_stock_list = [hmax_per_stock_list[0] if hmax_per_stock_list else 1000] * num_stocks_env

        tech_indicators = self.config.get('tech_indicator_list', [])
        lookback = self.config.get('lookback_window', 30)
        reward_scale = self.config.get('reward_scaling', 1e-4)
        env_seed = self.config.get('random_seed', None)

        env_metrics_path = os.path.join(self.output_dir, 'td3_backtest_env_metrics.csv')

        # Create and run backtest environment
        backtest_env = MultiStockTradingEnv(
            df=df_backtest,
            num_stocks=num_stocks_env,
            config=self.config, # Pass the full config
            initial_amount=initial_amount,
            buy_cost_pct=buy_cost_pct_list,
            sell_cost_pct=sell_cost_pct_list,
            hmax_per_stock=hmax_per_stock_list,
            reward_scaling=reward_scale,
            tech_indicator_list=tech_indicators,
            lookback_window=lookback,
            training=False, # Explicitly set to False for backtesting
            metrics_save_path=env_metrics_path, # Pass path for env's own metrics
            seed=env_seed
        )

        # Load model and normalize environment if env_path (for VecNormalize stats) is provided
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