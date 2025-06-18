import gymnasium as gym
import numpy as np
import pandas as pd
import os
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MultiStockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        num_stocks: int, # Typically from len(ticker_list) or derived from data
        config: dict, # Centralized configuration
        initial_amount: float = None, # Allow override from config
        buy_cost_pct: list[float] = None, # Allow override
        sell_cost_pct: list[float] = None, # Allow override
        hmax_per_stock: list[int] = None, # Allow override
        reward_scaling: float = None, # Allow override
        tech_indicator_list: list[str] = None, # Allow override
        risk_penalty: float = None, # Allow override (though might be in reward_params)
        lookback_window: int = None, # Allow override
        make_plots: bool = False, # Kept for direct control if needed
        print_verbosity: int = 10, # Kept for direct control
        stop_loss_threshold: float = None, # Allow override
        max_steps_per_episode: int = None, # Allow override
        training: bool = True,
        reward_params_override=None, # For specific overrides to reward_params from config
        metrics_save_path: str = None, # Explicit path for metrics CSV
        seed: int = None
    ):
        super().__init__()
        self.config = config
        self.seed = seed if seed is not None else self.config.get('random_seed', None) # Get seed from config if not passed

        self.num_stocks = num_stocks

        # Prioritize passed arguments, then config, then defaults
        self.initial_amount = initial_amount if initial_amount is not None else self.config.get('initial_amount', 100000.0)

        buy_cost_pct_default = [self.config.get('buy_cost_pct', 0.001)] * self.num_stocks if isinstance(self.config.get('buy_cost_pct', 0.001), float) else self.config.get('buy_cost_pct', [0.001]*self.num_stocks)
        self.buy_cost_pct = np.array(buy_cost_pct if buy_cost_pct is not None else buy_cost_pct_default)

        sell_cost_pct_default = [self.config.get('sell_cost_pct', 0.001)] * self.num_stocks if isinstance(self.config.get('sell_cost_pct', 0.001), float) else self.config.get('sell_cost_pct', [0.001]*self.num_stocks)
        self.sell_cost_pct = np.array(sell_cost_pct if sell_cost_pct is not None else sell_cost_pct_default)

        hmax_default = [self.config.get('hmax_per_stock', 1000)] * self.num_stocks if isinstance(self.config.get('hmax_per_stock', 1000), (int,float)) else self.config.get('hmax_per_stock', [1000]*self.num_stocks)
        self.hmax_per_stock = np.array(hmax_per_stock if hmax_per_stock is not None else hmax_default)

        self.reward_scaling = reward_scaling if reward_scaling is not None else self.config.get('reward_scaling', 1e-4)
        self.tech_indicator_list = tech_indicator_list if tech_indicator_list is not None else self.config.get('tech_indicator_list', [])
        self.lookback_window = lookback_window if lookback_window is not None else self.config.get('lookback_window', 30)
        self.stop_loss_threshold = stop_loss_threshold if stop_loss_threshold is not None else self.config.get('stop_loss_threshold', 0.15)

        # Risk penalty might be part of reward_params in config or a general env param
        self.risk_penalty = risk_penalty if risk_penalty is not None else self.config.get('risk_penalty', 0.0005)

        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.max_steps_per_episode = max_steps_per_episode if max_steps_per_episode is not None else self.config.get('max_steps_per_episode', 2000)
        self.training = training

        # Initialize reward_params: defaults < config < override
        base_reward_params = { # These are the original hardcoded defaults
            'k_p': 2.0, 'k_l': 0.5, 'gamma': 4.0, 'alpha': 1.5, 'beta': 1.5, 'lambda': 0.5,
            'lookback_window': self.lookback_window, 'w_risk': 0.1, 'w_drawdown': 0.1, 'w_action': 0.05,
            'phi': [0.05, 0.05, 0.05, 0.05], 'epsilon': 0.05, 'weight_min': 0.05, 'k_a': 0.05,
            'eta': 0.3, 'r_threshold': 0.05, 'reward_scaling_factor': 1e2, 'k_d': 0.2,
            'delta': 1.2, 'rho': 0.1, 'k_r': 0.1, 'debug': False,
            'fixed_reward_weights': {'w_profit': 0.7, 'w_risk': 0.1, 'w_drawdown': 0.1, 'w_action': 0.1} # For non-adaptive
        }
        self.reward_params = base_reward_params
        self.reward_params.update(self.config.get('reward_params', {}))
        if reward_params_override:
            self.reward_params.update(reward_params_override)

        # Determine actual_metrics_save_path
        if metrics_save_path:
            self.actual_metrics_save_path = metrics_save_path
        else:
            log_dir = self.config.get('log_dir', 'logs') # Default log_dir if not in config
            metrics_filename = self.config.get('trading_metrics_filename', 'trading_environment_metrics.csv')
            # PROJECT_ROOT should be in config from config_loader
            project_root = self.config.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            self.actual_metrics_save_path = os.path.join(project_root, log_dir, metrics_filename)

        self.df = self._preprocess_data(df.copy())
        self.df_info = pd.DataFrame() # Initialize df_info for storing step metrics
        # print(self.df.head()) # Reduced verbosity, can be enabled by debug flags
        self._validate_data()
        self._initialize_scalers()
        self._setup_spaces()
        self.reset()

    def _preprocess_data(self, df):
        for stock_id in range(self.num_stocks):
            close_col = f'close_{stock_id}'
            volume_col = f'volume_{stock_id}'
            
            # Add momentum and volume trends
            df[f'momentum_{stock_id}'] = df[close_col].diff(10)
            df[f'volume_trend_{stock_id}'] = df[volume_col].diff(5)
            
        # Fill NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df

    def _validate_data(self):
        required_columns = ["snp500", "gold_price", "interest_rate"]
        for stock in range(self.num_stocks):
            required_columns.extend([
                f"open_{stock}", f"high_{stock}", f"low_{stock}",
                f"close_{stock}", f"volume_{stock}", 
                f"eps_{stock}", f"pe_ratio_{stock}", f"volatility_30d_{stock}",
                f"momentum_{stock}", f"volume_trend_{stock}"
            ])
            required_columns.extend([f"{tech}_{stock}" for tech in self.tech_indicator_list])
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        assert not missing_columns, f"Missing required columns in dataframe: {missing_columns}"
        
    def _initialize_scalers(self):
        self.price_scaler = StandardScaler()
        self.tech_scaler = StandardScaler()
        self.macro_scaler = StandardScaler()
        
        # Fit scalers
        price_cols = []
        tech_cols = []
        for stock in range(self.num_stocks):
            price_cols.extend([f'open_{stock}', f'high_{stock}', f'low_{stock}', f'close_{stock}', f'volume_{stock}'])
            tech_cols.extend([
                f'eps_{stock}', f'pe_ratio_{stock}', f'volatility_30d_{stock}',
                f'momentum_{stock}', f'volume_trend_{stock}'
            ])
            tech_cols.extend([f'{tech}_{stock}' for tech in self.tech_indicator_list])
        
        self.price_scaler.fit(self.df[price_cols].values)
        self.tech_scaler.fit(self.df[tech_cols].values)
        self.macro_scaler.fit(self.df[['snp500', 'gold_price', 'interest_rate']].values)

    def _setup_spaces(self):
        # Action space: [-1, 1] for each stock, where:
        # -1 = sell all shares, 1 = buy with all available cash
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_stocks,), 
            dtype=np.float32
        )

        # Observation space components
        cash_shares_length = 1 + self.num_stocks  # cash + shares per stock
        price_features = 5 * self.num_stocks      # OHLCV per stock
        tech_features = (3 + 2 + len(self.tech_indicator_list)) * self.num_stocks  # earnings, pe, volatility, momentum, volume_trend + tech
        macro_features = 3                        # snp500, gold, inflation, rates
        
        total_features = cash_shares_length + price_features + tech_features + macro_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_features,), 
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.seed = seed or self.seed
        np.random.seed(self.seed)
        
        self.weights = self.initialize_weights([1.0, 1.0, 1.0, 1.0])
        
        self.current_step = 0
        self.cash = self.initial_amount
        self.shares = np.zeros(self.num_stocks, dtype=np.float32) # Ensure float for precision
        self.portfolio_value = float(self.initial_amount)        
        self.peak_value = self.initial_amount
        self.vwap = np.zeros(self.num_stocks, dtype=np.float32) # Ensure float
        
        self.trading_actions = []
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.daily_returns = [] # List to store daily/step returns
        self.cost = 0.0 # Ensure float
        self.trades = 0
        self.total_trades = 0 # Persists across resets if not reset here
        self.drawdown = 0.0 # Ensure float
        self.max_drawdown = 0.0 # Ensure float
        
        if self.training and self.max_steps_per_episode is not None:
            self.max_steps = self.max_steps_per_episode
        else:
            # For eval or if max_steps_per_episode not set, run through entire DataFrame
            self.max_steps = len(self.df) - 1

        self.data = self.df.iloc[self.current_step]
        # Initialize self.weights using reward_params for non-adaptive setup
        fixed_weights = self.reward_params.get('fixed_reward_weights', {'w_profit': 0.7, 'w_risk': 0.1, 'w_drawdown': 0.1, 'w_action': 0.1})
        self.weights = self.initialize_weights([
            fixed_weights.get('w_profit', 0.7),
            fixed_weights.get('w_risk', 0.1),
            fixed_weights.get('w_drawdown', 0.1),
            fixed_weights.get('w_action', 0.1)
        ])
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize cash and shares
        cash_normalized = np.array([self.cash / self.initial_amount], dtype=np.float32)
        # Avoid division by zero if hmax_per_stock can be zero, though unlikely by design
        hmax_safe = np.where(self.hmax_per_stock == 0, 1, self.hmax_per_stock)
        shares_normalized = self.shares / hmax_safe
        
        # Price features
        price_data = []
        for i in range(self.num_stocks):
            price_data.extend([
                self.data[f'open_{i}'], self.data[f'high_{i}'],
                self.data[f'low_{i}'], self.data[f'close_{i}'],
                self.data[f'volume_{i}']
            ])
        scaled_prices = self.price_scaler.transform([price_data])[0]
        
        # Technical features
        tech_data = []
        for i in range(self.num_stocks):
            tech_data.extend([
                self.data[f'eps_{i}'], self.data[f'pe_ratio_{i}'],
                self.data[f'volatility_30d_{i}'], self.data[f'momentum_{i}'],
                self.data[f'volume_trend_{i}']
            ])
            tech_data.extend([self.data[f'{tech}_{i}'] for tech in self.tech_indicator_list])
        scaled_tech = self.tech_scaler.transform([tech_data])[0]
        
        # Macro features
        macro_data = [
            self.data['snp500'], self.data['gold_price'],
             self.data['interest_rate']
        ]
        scaled_macro = self.macro_scaler.transform([macro_data])[0]
        
        # Concatenate all features
        obs = np.concatenate([
            cash_normalized,
            shares_normalized,
            scaled_prices,
            scaled_tech,
            scaled_macro
        ]).astype(np.float32)
        
        return obs

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.trading_actions.append(action)
        prev_value = self.portfolio_value

        self._execute_trades(action)

        terminated = False # Standard Gymnasium variable
        truncated = False  # Standard Gymnasium variable

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True # Episode truncated due to reaching max steps

        if not truncated:
            self.data = self.df.iloc[self.current_step]
            self._update_portfolio_value()
            if self._check_stop_loss(): # If stop loss triggered for all, could be terminal
                # A more robust terminal condition might be if cash <=0 and no shares.
                pass # For now, stop loss just sells, doesn't terminate episode.
            if self.portfolio_value <= self.initial_amount * 0.1: # Example: bankrupt condition
                terminated = True

        if terminated or truncated:
            # Save metrics at the end of the episode
            try:
                os.makedirs(os.path.dirname(self.actual_metrics_save_path), exist_ok=True)
                self.df_info.to_csv(self.actual_metrics_save_path, mode='a', header=not os.path.exists(self.actual_metrics_save_path), index=False)
            except IOError as e:
                print(f"Error saving trading metrics to {self.actual_metrics_save_path}: {e}")
            except Exception as e: # Catch other potential errors during save
                print(f"Unexpected error saving trading metrics: {e}")

        # Calculate portfolio return and update daily_returns
        portfolio_return = (self.portfolio_value - prev_value) / prev_value if prev_value != 0 else 0.0
        # daily_return is the same as portfolio_return for single step
        self.daily_returns.append(portfolio_return)

        # Reward calculation now uses self.reward_params internally
        reward = self._calculate_reward(prev_value, portfolio_return, action)

        # update_weights is now non-adaptive or uses fixed weights from self.reward_params
        self.update_weights() # Simplified call

        scaled_reward = reward * self.reward_scaling # self.reward_scaling from __init__/config
        self.asset_memory.append(self.portfolio_value)
        self.rewards_memory.append(scaled_reward)

        info = {
        'portfolio_value': self.portfolio_value,
        'reward': scaled_reward,
        'max_drawdown': self.max_drawdown,
        'trades': self.trades,
        'cost': self.cost,
        'drawdown': self.drawdown,
        'cumulative_return': (self.portfolio_value - self.initial_amount) / self.initial_amount,
        'volatility': np.std(self.daily_returns[-self.reward_params['lookback_window']:]) if len(self.daily_returns) > 0 else 0,
        'total_trades': self.total_trades,
        'average_trade_return': np.mean(self.daily_returns) if len(self.daily_returns) > 0 else 0,
        'current_holdings': self.shares.tolist(),
            'current_prices': [self.data[f'close_{i}'] for i in range(self.num_stocks)] if not (terminated or truncated) else [0]*self.num_stocks,
        'vwap': self.vwap.tolist(),
        'action_taken': action.tolist(),
            'current_date': self.df.index[self.current_step] if self.current_step < len(self.df.index) else "EndOfData",
        }
        
        # Append step info to df_info
        # Ensure all values in info are serializable if df_info is used broadly
        try:
            self.df_info = pd.concat([self.df_info, pd.DataFrame([info])], ignore_index=True)
        except Exception as e:
            print(f"Error concatenating info to df_info: {e}. Info: {info}")

        # print(info) # Controlled by print_verbosity or debug flags in config
        if self.reward_params.get('debug', False) and self.current_step % self.print_verbosity == 0 :
             print(f"Step: {self.current_step}, Action: {action}, Reward: {scaled_reward}, Portfolio: {self.portfolio_value}")

        return self._get_obs(), scaled_reward, terminated, truncated, info

    def _execute_trades(self, action):
        # action is assumed to be normalized between -1 and 1 for each stock
        # -1: sell max, 0: hold, 1: buy max
        for stock_id in range(self.num_stocks):
            action_val = action[stock_id]
            
            if action_val < 0:  # Sell
                self._sell_stock(stock_id, abs(action_val))
            elif action_val > 0:  # Buy
                self._buy_stock(stock_id, action_val)

    def _sell_stock(self, stock_id, sell_pct):
        sell_pct = max(0.0, min(sell_pct, 1.0))
        available_shares = self.shares[stock_id]
        
        shares_to_sell = int(available_shares * sell_pct)
        
        shares_to_sell = min(shares_to_sell, self.hmax_per_stock[stock_id])
        
        if shares_to_sell > 0:
            price = self.data[f'close_{stock_id}']
            cost_pct = self.sell_cost_pct[stock_id]
            
            proceeds = shares_to_sell * price * (1 - cost_pct)
            self.cash += proceeds
            self.shares[stock_id] -= shares_to_sell
            self.cost += shares_to_sell * price * cost_pct
            self.trades += 1
            self.total_trades += 1
            
            if self.shares[stock_id] == 0:
                self.vwap[stock_id] = 0.0


    def _buy_stock(self, stock_id, buy_pct):
        buy_pct = max(0.0, min(buy_pct, 1.0))
        available_cash = self.cash * buy_pct
        price = self.data[f'close_{stock_id}']
        cost_pct = self.buy_cost_pct[stock_id]
        
        if available_cash > 0 and price > 0:
            
            max_affordable = available_cash / (price * (1 + cost_pct))
            max_shares = min(max_affordable, self.hmax_per_stock[stock_id])
            shares_to_buy = int(max_shares)
            
            if shares_to_buy > 0:
                total_cost = shares_to_buy * price * (1 + cost_pct)
                self.cash -= total_cost
                self.shares[stock_id] += shares_to_buy
                self.cost += total_cost - (shares_to_buy * price)
                self.trades += 1
                self.total_trades += 1
                
                total_value = self.vwap[stock_id] * (self.shares[stock_id] - shares_to_buy) + total_cost
                self.vwap[stock_id] = total_value / self.shares[stock_id] if self.shares[stock_id] > 0 else 0

    def _update_portfolio_value(self):
        stock_values = self.shares * np.array([self.data[f'close_{i}'] for i in range(self.num_stocks)])
        self.portfolio_value = self.cash + np.sum(stock_values)
        
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        self.drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, self.drawdown)

    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_return: float, action: np.ndarray):
        """
        Calculate reward based on portfolio performance and risk.
        Uses self.reward_params for all parameters.
        """
        # Use self.weights which are now set from fixed_reward_weights in reward_params or defaults
        weights = self.weights

        # Smoothed return over a lookback window from self.reward_params
        lookback = self.reward_params.get('lookback_window', self.lookback_window)
        # Ensure daily_returns has enough data for mean calculation
        smoothed_return = np.mean(self.daily_returns[-lookback:]) if len(self.daily_returns) >= lookback else current_portfolio_return
        
        f_profit = self.calculate_profit_term(smoothed_return) # Pass only necessary arg

        penalty_risk = self.calculate_risk_penalty() # Uses self.daily_returns, self.portfolio_value
        penalty_drawdown = self.calculate_drawdown_penalty() # Uses self.drawdown, self.portfolio_value
        penalty_action = self.calculate_action_penalty(current_portfolio_return, action)

        # Scale down penalties to reduce their impact - this logic can be part of reward_params
        penalty_scaling = self.reward_params.get('penalty_scaling_factor', 0.2)
        max_total_penalty = self.reward_params.get('max_total_penalty_clamp', 0.2)

        total_penalty = penalty_scaling * (penalty_risk + penalty_drawdown + penalty_action)
        total_penalty = min(total_penalty, max_total_penalty)

        volatility = np.std(self.daily_returns[-lookback:]) if len(self.daily_returns) >= lookback else 1e-6
        sharpe_ratio_component = self.reward_params.get('sharpe_ratio_weight', 0.5) * (smoothed_return / (volatility + 1e-6))

        reward = (
            weights['w_profit'] * f_profit +
            sharpe_ratio_component - # Add risk-adjusted returns
            total_penalty # Subtract scaled penalties
        )

        # Final scaling factor for reward magnitude, from reward_params
        final_reward_scaling = self.reward_params.get('reward_scaling_factor', 1e2) # This is different from self.reward_scaling
        reward *= final_reward_scaling
  
        if self.reward_params.get('debug', False):
            print(f"CalcReward - Return: {current_portfolio_return:.4f}, SmoothRet: {smoothed_return:.4f}, "
                  f"f_profit: {f_profit:.4f}, P_risk: {penalty_risk:.4f}, P_drawdown: {penalty_drawdown:.4f}, "
                  f"P_action: {penalty_action:.4f}, TotalPenalty: {total_penalty:.4f}, SharpeComp: {sharpe_ratio_component:.4f}, "
                  f"FinalReward: {reward:.4f}")
        return reward

    def _check_stop_loss(self):
        triggered = False
        for stock_id in range(self.num_stocks):
            if self.shares[stock_id] > 0 and self.vwap[stock_id] > 0: # Check if VWAP is valid
                current_price = self.data[f'close_{stock_id}']
                loss_pct = (current_price - self.vwap[stock_id]) / self.vwap[stock_id]
                if loss_pct < -self.stop_loss_threshold: # self.stop_loss_threshold from __init__/config
                    self._sell_stock(stock_id, 1.0)  # Sell all shares of this stock
                    triggered = True
        return triggered # Return if any stop loss was triggered

    def render(self, mode="human"): # mode argument is standard for gym.Env
        if self.current_step % self.print_verbosity == 0 and self.print_verbosity > 0:
            profit = self.portfolio_value - self.initial_amount
            print(f"Step: {self.current_step}")
            print(f"Value: ${self.portfolio_value:.2f} | Profit: ${profit:.2f}")
            print(f"Cash: ${self.cash:.2f} | Shares: {self.shares}")
            print(f"Drawdown: {self.drawdown:.2%} | Trades: {self.trades}")
            
        
    def initialize_weights(self,initial_weights):
   
        w_profit, w_risk, w_drawdown, w_action = initial_weights
        weight_min = 0.01  # Lower minimum threshold for weights
        epsilon = 0.01  # Smaller clip value for weight updates
        
        return {
            'w_profit': w_profit,
            'w_risk': w_risk, # Weight for risk penalty component
            'w_drawdown': w_drawdown, # Weight for drawdown penalty component
            'w_action': w_action, # Weight for action penalty component
            'weight_min': self.reward_params.get('weight_min', 0.01), # From central reward_params
            'epsilon': self.reward_params.get('epsilon', 0.01) # From central reward_params
        }

    def calculate_profit_term(self, portfolio_return):
        # Uses self.reward_params
        base_reward_multiplier = self.reward_params.get('base_reward_multiplier', 10)
        k_p = self.reward_params.get('k_p', 2.0) * 2.0 # Example: Amplify from base config
        k_l = self.reward_params.get('k_l', 0.5) * 1
        
        scaled_return = portfolio_return * 100 # Scale for calculation
        
        if scaled_return > 0:
            f_profit = k_p * abs(scaled_return)
        else:
            f_profit = -k_l * abs(scaled_return)
        
        f_profit *= base_reward_multiplier
        f_profit = np.clip(f_profit, -100, 100) # Clip to a reasonable range
        return f_profit

    def calculate_risk_penalty(self):
        # Uses self.reward_params, self.daily_returns, self.portfolio_value
        k_r = self.reward_params.get('k_r', 0.1) * 0.5
        beta = self.reward_params.get('beta', 1.5)
        lambda_ = self.reward_params.get('lambda', 0.5) # Underscore to avoid keyword conflict
        lookback = self.reward_params.get('lookback_window', self.lookback_window)
        w_risk = self.reward_params.get('w_risk', 0.1)

        returns_window = np.array(self.daily_returns[-lookback:])
        volatility = np.std(returns_window) if len(returns_window) >= lookback else 0.0
        avg_return = np.mean(returns_window) if len(returns_window) >= lookback else 0.0

        # Softplus to ensure denominator is positive and smooth
        # The term (avg_return - volatility) could be negative, lambda_ scales its effect
        exponent_val = lambda_ * (avg_return - volatility)
        exponent_val = np.clip(exponent_val, -50, 50) # Avoid overflow in exp
        softplus_risk_denominator = np.log(1 + np.exp(exponent_val)) + 1e-6 # Add epsilon for stability

        penalty_risk = w_risk * (k_r * (volatility ** beta) / softplus_risk_denominator)
        return penalty_risk

    def calculate_drawdown_penalty(self):
        # Uses self.reward_params, self.drawdown, self.portfolio_value
        k_d = self.reward_params.get('k_d', 0.2) * 1.5
        delta = self.reward_params.get('delta', 1.2)
        rho = self.reward_params.get('rho', 0.1) * 0.5
        w_drawdown = self.reward_params.get('w_drawdown', 0.1)

        # Softplus for drawdown effect related to portfolio value
        # The term calculates how far the current value is from peak, scaled by rho
        exponent_val = rho * (self.peak_value - self.portfolio_value) # self.peak_value - self.portfolio_value is the absolute drawdown amount
        exponent_val = np.clip(exponent_val, -50, 50)
        softplus_drawdown_denominator = np.log(1 + np.exp(exponent_val)) + 1e-6

        penalty_drawdown = w_drawdown * (k_d * (self.drawdown ** delta) / softplus_drawdown_denominator)
        return penalty_drawdown

    def calculate_action_penalty(self, portfolio_return, action_vector):
        # Uses self.reward_params
        k_a = self.reward_params.get('k_a', 0.05)
        eta = self.reward_params.get('eta', 0.3)
        r_threshold = self.reward_params.get('r_threshold', 0.05)
        w_action = self.reward_params.get('w_action', 0.05)
        
        penalty_action = 0.0
        # Penalize actions more if portfolio return is small (indecisive or churning)
        if abs(portfolio_return) < r_threshold:
            # L2 norm of action vector to penalize large actions
            penalty_action = w_action * (k_a * np.linalg.norm(action_vector)**2 * np.exp(-eta * abs(portfolio_return)))
        
        return penalty_action

    # calculate_total_penalty and the second calculate_reward are removed as their logic
    # is integrated into the main _calculate_reward method.

    def update_weights(self):
        """
        Simplified to use fixed weights from reward_params or default initial weights.
        The adaptive logic is removed as per subtask requirement for now.
        self.weights is initialized in reset() and used by _calculate_reward().
        """
        # This method could be used in the future to re-load weights if they were
        # made configurable per-step or per-episode from some external source.
        # For now, it does nothing as weights are fixed per episode after reset.
        pass
        # Example if we wanted to allow dynamic reloading of fixed weights from self.reward_params:
        # fixed_weights_config = self.reward_params.get('fixed_reward_weights', {'w_profit': 0.7, 'w_risk': 0.1, 'w_drawdown': 0.1, 'w_action': 0.1})
        # self.weights = self.initialize_weights([
        #     fixed_weights_config.get('w_profit', 0.7),
        #     fixed_weights_config.get('w_risk', 0.1),
        #     fixed_weights_config.get('w_drawdown', 0.1),
        #     fixed_weights_config.get('w_action', 0.1)
        # ])