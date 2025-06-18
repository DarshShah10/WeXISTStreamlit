import os
import pandas as pd
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from src.envs.trading_env import MultiStockTradingEnv


class TD3TradingBot:
    def __init__(self, num_stocks=5, initial_amount=100000.00):
        self.num_stocks = num_stocks
        self.initial_amount = initial_amount
        self.env = None
        self.model = None

    def load_data(self, csv_file):
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["Date"])
        df.set_index("date", inplace=True)
        return df

    def make_env(self, df, training=True):
        return MultiStockTradingEnv(
            df=df,
            num_stocks=self.num_stocks,
            initial_amount=self.initial_amount,
            buy_cost_pct=[0.001] * self.num_stocks,
            sell_cost_pct=[0.001] * self.num_stocks,
            hmax_per_stock=[1000] * self.num_stocks,
            reward_scaling=1e-4,
            tech_indicator_list=[
                "sma50", "sma200", "ema12", "ema26", "macd", "rsi", "cci", "adx",
                "sok", "sod", "du", "dl", "vm", "bb_upper", "bb_lower", "bb_middle", "obv"
            ],
            lookback_window=30,
            training=training,
            max_steps_per_episode=250,
        )

    def train_env(self, df):
        env = DummyVecEnv([lambda: self.make_env(df)])
        return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    def train_from_scratch(self, csv_file_path, total_timesteps=100000, save_path="./models/"):
        
        print("Starting training from scratch...")
        
        # Create save directories
        os.makedirs(save_path, exist_ok=True)
        checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load and prepare environment
        df = self.load_data(csv_file_path)
        self.env = self.train_env(df)
        
        # Initialize new TD3 model
        self.model = TD3(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=128,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            action_noise=NormalActionNoise(
                mean=np.zeros(self.num_stocks), 
                sigma=0.1 * np.ones(self.num_stocks)
            ),
            policy_kwargs=dict(net_arch=[400, 300]),
        )
        
        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,  # Save every 10000 steps
            save_path=checkpoint_dir,
            name_prefix="td3_trading",
            save_vecnormalize=True  # This will save the environment along with the model
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model and environment
        final_model_path = os.path.join(save_path, "td3_stock_trading_final")
        final_env_path = os.path.join(save_path, "td3_stock_trading_env_final.pkl")
        
        self.model.save(final_model_path)
        self.env.save(final_env_path)
        
        # Evaluate the model
        self.evaluate_model()
        
        return final_model_path, final_env_path

    def train_pretrained(self, csv_file_path, model_path, env_path, total_timesteps=50000, save_path="./models/"):
        
        print("Loading pretrained model and continuing training...")
        
        # Create save directories
        os.makedirs(save_path, exist_ok=True)
        checkpoint_dir = os.path.join(save_path, "checkpoints_continued")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load and prepare environment
        df = self.load_data(csv_file_path)
        self.env = VecNormalize.load(env_path, DummyVecEnv([lambda: self.make_env(df)]))
        self.env.training = True
        self.env.norm_reward = True
        
        # Load pretrained model
        self.model = TD3.load(model_path, env=self.env)
        
        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,  # Save every 10000 steps
            save_path=checkpoint_dir,
            name_prefix="td3_trading_continued",
            save_vecnormalize=True  # This will save the environment along with the model
        )
        
        # Continue training
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        # Save final model and environment
        final_model_path = os.path.join(save_path, "td3_stock_trading_final")
        final_env_path = os.path.join(save_path, "td3_stock_trading_env_final.pkl")
        
        self.model.save(final_model_path)
        self.env.save(final_env_path)
        
        # Evaluate the model
        self.evaluate_model()
        
        return final_model_path, final_env_path

    def evaluate_model(self, n_eval_episodes=10):
       
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_eval_episodes)
        print(f"Mean reward: {mean_reward} ± {std_reward}")
        
        if hasattr(self.env, "save_metrics"):
            self.env.save_metrics()
        
        return mean_reward, std_reward
    
