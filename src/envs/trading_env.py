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
        num_stocks: int,
        initial_amount: float,
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        hmax_per_stock: list[int],
        reward_scaling: float,
        tech_indicator_list: list[str],
        risk_penalty: float = 0.0005,
        lookback_window: int = 30,
        make_plots: bool = False,
        print_verbosity: int = 10,
        stop_loss_threshold: float = 0.15,
        max_steps_per_episode: int = None,
        training: bool = True,
        reward_params=None,
        seed: int = None
    ):
        super().__init__()
        self.seed = seed
        self.num_stocks = num_stocks
        self.initial_amount = initial_amount
        self.buy_cost_pct = np.array(buy_cost_pct)
        self.sell_cost_pct = np.array(sell_cost_pct)
        self.hmax_per_stock = np.array(hmax_per_stock)
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.risk_penalty = risk_penalty
        self.lookback_window = lookback_window
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.max_steps_per_episode = max_steps_per_episode
        self.training = training
        self.stop_loss_threshold = stop_loss_threshold
        self.df = self._preprocess_data(df.copy())
        self.reward_params = reward_params or {
            'k_p': 2.0,  #  profit coefficient
            'k_l': 0.5,  #  loss coefficient
            'gamma': 4.0,  #  gamma ->  profit amplification
            'alpha': 1.5,  #  alpha -> sensitivity to losses
            'beta': 1.5,  #   beta -> soften risk penalty
            'lambda': 0.5,  #  lambda -> risk penalty
            'lookback_window': self.lookback_window or 30,
            'w_risk': 0.1,  #  weight for risk penalty
            'w_drawdown': 0.1,  #  weight for drawdown penalty
            'w_action': 0.05,  #  weight for action penalty
            'phi': [0.05, 0.05, 0.05, 0.05],  #  learning rates
            'epsilon': 0.05,
            'weight_min': 0.05,  #  minimum threshold for weights
            'k_a': 0.05,  #  action penalty coefficient
            'eta': 0.3,  #  eta to soften action penalty
            'r_threshold': 0.05,
            'reward_scaling_factor': 1e2,  #  scaling factor for larger rewards
            'k_d': 0.2,  #  drawdown penalty coefficient
            'delta': 1.2,  #  delta to soften drawdown penalty
            'rho': 0.1,  #  rho to relax drawdown penalty
            'k_r': 0.1,  #  risk penalty coefficient
            'debug': False
        }
        self.df_info = pd.DataFrame()
        print(self.df.head())
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
        self.shares = np.zeros(self.num_stocks)
        self.portfolio_value = float(self.initial_amount)        
        self.peak_value = self.initial_amount
        self.vwap = np.zeros(self.num_stocks)
        
        self.trading_actions = []
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.daily_returns = []
        self.cost = 0
        self.trades = 0
        self.total_trades = 0
        self.drawdown = 0
        self.max_drawdown = 0
        
        if self.training:
            self.max_steps = self.max_steps_per_episode or len(self.df) - 1
        else:
            self.max_steps = len(self.df) - 1

        self.data = self.df.iloc[self.current_step]
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize cash and shares
        cash_normalized = np.array([self.cash / self.initial_amount])
        shares_normalized = self.shares / self.hmax_per_stock
        
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
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if not done:
            self.data = self.df.iloc[self.current_step]
            self._update_portfolio_value()
            self._check_stop_loss()
        else:
            self.df_info.to_csv('trading_metrics.csv', mode='a', header=not os.path.exists('trading_metrics.csv'))
            # self.df_info.to_csv('trading_metrics.csv')
            

        # Calculate portfolio return and update daily_returns
        portfolio_return = (self.portfolio_value - prev_value) / prev_value if prev_value != 0 else 0
        daily_return = portfolio_return
        self.daily_returns.append(daily_return)

        # Collect variables for reward calculation
        drawdown = self.drawdown
        cost = self.cost
        shares = self.shares.copy()
        stock_prices = np.array([self.data[f'close_{i}'] for i in range(self.num_stocks)])
        action_vector = action.copy()

        # Calculate reward
        reward = self._calculate_reward(
            prev_value, 
            self.daily_returns, 
            drawdown, 
            cost, 
            shares, 
            stock_prices, 
            action_vector, 
            self.reward_params
        )

        # Update weights dynamically
        volatility = np.std(self.daily_returns[-self.reward_params['lookback_window']:])
        self.weights = self.update_weights(
            self.weights, 
            portfolio_return, 
            volatility, 
            drawdown, 
            action_vector, 
            self.reward_params
        )

        scaled_reward = reward * self.reward_scaling
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
        'current_prices': [self.data[f'close_{i}'] for i in range(self.num_stocks)],
        'vwap': self.vwap.tolist(),
        'action_taken': action.tolist(),
        'current_date': self.df.index[self.current_step],
        }
        
        self.df_info = pd.concat([self.df_info, pd.DataFrame([info])], ignore_index=True)
        print(info)
        return self._get_obs(), scaled_reward, done, False, info

    def _execute_trades(self, action):
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

    def _calculate_reward(self, prev_value, daily_returns, drawdown, cost, shares, stock_prices, action_vector, params):

        
        initial_weights = [0.7, 0.1, 0.1, 0.1]  # Focus on profit, reduce risk/drawdown/action penalties
        weights = self.initialize_weights(initial_weights)

        # Calculate portfolio return
        portfolio_return = (self.portfolio_value - prev_value) / prev_value if prev_value != 0 else 0

        # Smoothed return over a lookback window
        lookback_window = params['lookback_window']
        smoothed_portfolio_return = np.mean(daily_returns[-lookback_window:]) if len(
            daily_returns) > 0 else portfolio_return

        f_profit = self.calculate_profit_term(smoothed_portfolio_return, params)

        # Calculate penalties
        penalty_risk = self.calculate_risk_penalty(daily_returns, self.portfolio_value, params)
        penalty_drawdown = self.calculate_drawdown_penalty(drawdown, self.portfolio_value, params)
        penalty_action = self.calculate_action_penalty(portfolio_return, action_vector, params)

        # Scale down penalties to reduce their impact
        total_penalty = 0.2 * (penalty_risk + penalty_drawdown + penalty_action)  # Reduced penalty impact
        total_penalty = min(total_penalty, 0.2)  # Clamp penalties to a small range

        volatility = np.std(daily_returns[-lookback_window:]) if len(daily_returns) > 0 else 1e-6
        sharpe_ratio = smoothed_portfolio_return / (volatility + 1e-6)  # Avoid division by zero

        # Combine profit term, Sharpe ratio, and penalties
        reward = (
                weights['w_profit'] * f_profit +  # Amplified profit term
                0.5 * sharpe_ratio -  # Add risk-adjusted returns
                total_penalty  # Subtract scaled penalties
        )

   
        reward_scaling_factor = 1e2

        reward *= reward_scaling_factor
  
        # Debugging: Log intermediate values for analysis
        if params.get('debug', False):
            print(f"Portfolio Return: {portfolio_return:.6f}, Smoothed Return: {smoothed_portfolio_return:.6f}, "
                  f"Profit Term: {f_profit:.6f}, Risk Penalty: {penalty_risk:.6f}, Drawdown Penalty: {penalty_drawdown:.6f}, "
                  f"Action Penalty: {penalty_action:.6f}, Total Penalty: {total_penalty:.6f}, Sharpe Ratio: {sharpe_ratio:.6f}, "
                  f"Reward: {reward:.6f}")

        return reward

    def _check_stop_loss(self):
        for stock_id in range(self.num_stocks):
            if self.shares[stock_id] > 0 and self.vwap[stock_id] > 0:
                current_price = self.data[f'close_{stock_id}']
                loss_pct = (current_price - self.vwap[stock_id]) / self.vwap[stock_id]
                if loss_pct < -self.stop_loss_threshold:
                    self._sell_stock(stock_id, 1.0)  # Sell all shares

    def render(self, mode="human"):
        if self.current_step % self.print_verbosity == 0:
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
            'w_risk': w_risk,
            'w_drawdown': w_drawdown,
            'w_action': w_action,
            'weight_min': weight_min,
            'epsilon': epsilon
        }

    def calculate_profit_term(self, portfolio_return, params):
        
        base_reward_multiplier = params.get('base_reward_multiplier', 10)  
        k_p = params['k_p'] * 2.0  
        k_l = params['k_l'] * 1  
        
        scaled_return = portfolio_return * 100 
        
        # Calculate profit term based on scaled return
        if scaled_return > 0:
            f_profit = k_p * abs(scaled_return)  # Linear amplification for profits
        else:
            f_profit = -k_l * abs(scaled_return)  # Linear penalty for losses
        
        # Apply base reward multiplier
        f_profit *= 10
        
        f_profit = np.clip(f_profit, -100, 100)  
        
        return f_profit

    def calculate_risk_penalty(self, daily_returns, portfolio_value, params):
        
        k_r = params['k_r'] * 0.5  # Reduce risk penalty coefficient
        beta = params['beta']
        lambda_ = params['lambda']
        lookback_window = params['lookback_window']
        w_risk = params['w_risk']

        returns_window = np.array(daily_returns[-lookback_window:])
        volatility = np.std(returns_window) if len(returns_window) > 0 else 0
        avg_return = np.mean(returns_window) if len(returns_window) > 0 else 0
        softplus_risk = np.log(1 + np.exp(lambda_ * (avg_return - volatility)))
        penalty_risk = w_risk * (k_r * (volatility ** beta) / softplus_risk)

        return penalty_risk

    def calculate_drawdown_penalty(self, drawdown, portfolio_value, params):
             
        k_d = params['k_d'] * 1.5  # Increase drawdown penalty coefficient
        delta = params['delta']
        rho = params['rho'] * 0.5  # Adjust rho to balance the penalty
        w_drawdown = params['w_drawdown']

        exponent = rho * (portfolio_value - (portfolio_value * (1 - drawdown)))
        exponent = np.clip(exponent, -50, 50)  
        softplus_drawdown = np.log(1 + np.exp(exponent))

        penalty_drawdown = w_drawdown * (k_d * (drawdown ** delta) / softplus_drawdown)

        return penalty_drawdown

    def calculate_action_penalty(self , portfolio_return, action_vector, params):
        
        k_a = params['k_a']
        eta = params['eta']
        r_threshold = params['r_threshold']
        w_action = params['w_action']
        
        if abs(portfolio_return) < r_threshold:
            penalty_action = w_action * (k_a * np.linalg.norm(action_vector)**2 * np.exp(-eta * abs(portfolio_return)))
        else:
            penalty_action = 0
        
        return penalty_action


    def calculate_total_penalty(self,penalty_risk, penalty_drawdown, penalty_action):
    
        total_penalty = penalty_risk + penalty_drawdown + penalty_action
        total_penalty = min(total_penalty, 0.99)  # Clamp to prevent sign flip
        return total_penalty


    def calculate_reward(self,portfolio_return, total_penalty, f_profit, w_profit):
       
        reward = (
            np.sign(portfolio_return) * 
            w_profit * f_profit * 
            (1 - total_penalty)
        )
        return reward


    def update_weights(self,weights, portfolio_return, volatility, drawdown, action_vector, params):
         
        phi_profit, phi_risk, phi_drawdown, phi_action = [p * 0.5 for p in params['phi']]  # Reduce learning rates
        epsilon = weights['epsilon']
        weight_min = weights['weight_min']
        portfolio_value = params.get('portfolio_value', 1)  # Default to 1 if not provided
        
        # Differential updates
        dw_profit_dt = phi_profit * (portfolio_return / portfolio_value) if portfolio_value > 0 else 0
        dw_risk_dt = -phi_risk * volatility
        dw_drawdown_dt = -phi_drawdown * drawdown
        dw_action_dt = -phi_action * np.linalg.norm(action_vector)**2
        
        # Clip updates to prevent instability
        dw_profit_dt = np.clip(dw_profit_dt, -epsilon, epsilon)
        dw_risk_dt = np.clip(dw_risk_dt, -epsilon, epsilon)
        dw_drawdown_dt = np.clip(dw_drawdown_dt, -epsilon, epsilon)
        dw_action_dt = np.clip(dw_action_dt, -epsilon, epsilon)
        
        # Update weights
        weights['w_profit'] += dw_profit_dt
        weights['w_risk'] += dw_risk_dt
        weights['w_drawdown'] += dw_drawdown_dt
        weights['w_action'] += dw_action_dt
        
        # Ensure weights remain above minimum threshold
        weights['w_profit'] = max(weights['w_profit'], weight_min)
        weights['w_risk'] = max(weights['w_risk'], weight_min)
        weights['w_drawdown'] = max(weights['w_drawdown'], weight_min)
        weights['w_action'] = max(weights['w_action'], weight_min)
        
        return weights