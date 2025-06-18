import os
import pandas as pd
from datetime import datetime

from src.envs.New_Trading_Env import MultiStockTradingEnv
# Placeholder for data preprocessing pipeline - adjust import as per actual structure
# from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from stable_baselines3 import PPO, TD3, A2C  # Example agents
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np

# Configuration (move to a dedicated config file later if needed)
# --- Data Config ---
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "HDFC.NS", "KOTAKBANK.NS", "ITC.NS", "LT.NS",
    # Add more tickers as needed, or load from a file
]
START_DATE = "2018-01-01" # Reduced date range for faster example runs
END_DATE = "2021-01-01"
# Path to preprocessed data, IF data preprocessing is run separately
# PREPROCESSED_DATA_PATH = "path/to/your/combined_data.csv"

# --- Environment Config ---
NUM_STOCKS = len(TICKERS) # Should match the number of stocks in the preprocessed data
INITIAL_AMOUNT = 1000000.0
BUY_COST_PCT = [0.001] * NUM_STOCKS # Example: 0.1% per trade
SELL_COST_PCT = [0.001] * NUM_STOCKS # Example: 0.1% per trade
HMAX_PER_STOCK = [100] * NUM_STOCKS # Max shares to trade per stock
REWARD_SCALING = 1e-4 # Keep rewards in a reasonable range for the agent
# TECH_INDICATOR_LIST will depend on the data preprocessing step.
# Example, if these are columns in your preprocessed CSV:
TECH_INDICATOR_LIST = ["macd", "rsi_30", "cci_30", "dx_30"] # Adjust as per your data
STOP_LOSS_THRESHOLD = 0.10 # 10% stop loss
MAX_STEPS_PER_EPISODE = 2000 # Or length of your typical data segment for one episode

# --- Agent and Training Config ---
MODEL_NAME = "PPO" # PPO, TD3, A2C
TOTAL_TRAINING_TIMESTEPS = 100000 # Adjust for desired training length
LOG_INTERVAL = 1
CHECKPOINT_FREQ = 50000 # Save a checkpoint every N steps
EVAL_FREQ = 20000 # Evaluate the model every N steps
N_EVAL_EPISODES = 5 # Number of episodes for evaluation
LOG_DIR = "./logs/"
MODEL_SAVE_DIR = "./trained_models/"
SEED = 42

# --- Helper Functions ---
def create_env(df, training=True):
    """Helper function to create and wrap the environment."""
    env = MultiStockTradingEnv(
        df=df,
        num_stocks=NUM_STOCKS,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=BUY_COST_PCT,
        sell_cost_pct=SELL_COST_PCT,
        hmax_per_stock=HMAX_PER_STOCK,
        reward_scaling=REWARD_SCALING,
        tech_indicator_list=TECH_INDICATOR_LIST, # Make sure these columns exist in df
        stop_loss_threshold=STOP_LOSS_THRESHOLD,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        training=training, # Differentiates training and eval/test behavior
        seed=SEED
    )
    env = Monitor(env, LOG_DIR if training else None) # Log training stats
    return env

def train_agent(train_df, eval_df=None):
    """Trains the RL agent."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Create training environment
    # For SB3, it's common to wrap the environment in a DummyVecEnv for single envs
    train_env = DummyVecEnv([lambda: create_env(train_df, training=True)])
    train_env.seed(SEED) # Seed the vectorized environment

    # Create evaluation environment if eval_df is provided
    eval_env = None
    if eval_df is not None:
        eval_env = DummyVecEnv([lambda: create_env(eval_df, training=False)])
        eval_env.seed(SEED)


    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_SAVE_DIR,
        name_prefix=f"{MODEL_NAME.lower()}_stock_trader"
    )
    eval_callback = None
    if eval_env:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=MODEL_SAVE_DIR,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )

    callbacks = [checkpoint_callback]
    if eval_callback:
        callbacks.append(eval_callback)

    # Select and instantiate the agent
    if MODEL_NAME == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR, seed=SEED, n_steps=1024)
    elif MODEL_NAME == "A2C":
        model = A2C("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR, seed=SEED)
    elif MODEL_NAME == "TD3":
        # TD3 requires a continuous action space, ensure New_Trading_Env has this.
        # It also benefits from action noise.
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", train_env, action_noise=action_noise, verbose=1, tensorboard_log=LOG_DIR, seed=SEED)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    model.learn(
        total_timesteps=TOTAL_TRAINING_TIMESTEPS,
        log_interval=LOG_INTERVAL,
        callback=callbacks
    )

    # Save the final model
    final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME.lower()}_stock_trader_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to: {final_model_path}")
    return model

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting main script...")

    # --- 1. Data Preprocessing ---
    # This is a placeholder. In a real scenario, you would:
    # a) Run your data preprocessing pipeline (e.g., StockPreProcessPipeline)
    # b) Or load already preprocessed data.
    # For this example, we'll create a dummy DataFrame.
    # The columns must match what MultiStockTradingEnv expects.
    print("Simulating data preprocessing...")

    # Create dummy data that matches the expected structure after preprocessing
    # This needs to be significantly more robust for real use.
    num_days = MAX_STEPS_PER_EPISODE + 100 # Some buffer

    # Create a DatetimeIndex
    date_rng = pd.date_range(start=START_DATE, end=END_DATE, freq='B') # Business days
    if len(date_rng) < num_days:
        # If date range is too short, generate enough days from start_date
        date_rng = pd.date_range(start=START_DATE, periods=num_days, freq='B')
    else:
        date_rng = date_rng[:num_days]


    data = {'date': date_rng}

    # Macro features (same for all stocks on a given day)
    data['snp500'] = np.random.rand(num_days) * 1000 + 3000
    data['gold_price'] = np.random.rand(num_days) * 500 + 1500
    data['interest_rate'] = np.random.rand(num_days) * 0.05

    for i in range(NUM_STOCKS):
        data[f'open_{i}'] = np.random.rand(num_days) * 100 + 50
        data[f'high_{i}'] = data[f'open_{i}'] + np.random.rand(num_days) * 10
        data[f'low_{i}'] = data[f'open_{i}'] - np.random.rand(num_days) * 10
        data[f'close_{i}'] = (data[f'high_{i}'] + data[f'low_{i}']) / 2
        data[f'volume_{i}'] = np.random.randint(100000, 1000000, size=num_days)
        data[f'eps_{i}'] = np.random.rand(num_days) * 5 + 1
        data[f'pe_ratio_{i}'] = data[f'close_{i}'] / data[f'eps_{i}']
        data[f'volatility_30d_{i}'] = np.random.rand(num_days) * 0.02 + 0.01
        data[f'momentum_{i}'] = np.random.rand(num_days) * 20 - 10 # price change over 10 days
        data[f'volume_trend_{i}'] = np.random.rand(num_days) * 10000 - 5000 # volume change over 5 days

        # Add technical indicators based on TECH_INDICATOR_LIST
        for ti in TECH_INDICATOR_LIST:
            data[f'{ti}_{i}'] = np.random.rand(num_days) # Placeholder

    dummy_df = pd.DataFrame(data)
    dummy_df.set_index('date', inplace=True)

    # Fill NaNs just in case (though random generation shouldn't produce them here)
    dummy_df.fillna(method='ffill', inplace=True)
    dummy_df.fillna(method='bfill', inplace=True)

    # For simplicity, use the same df for training and evaluation here.
    # In practice, you MUST split your data into train/validation/test sets.
    # E.g., train_df = dummy_df[:'2020-06-01']
    #       eval_df  = dummy_df['2020-06-01':]
    train_df = dummy_df
    eval_df = dummy_df # Replace with actual split data

    print(f"Dummy DataFrame created with {len(train_df)} rows.")
    print(f"Columns: {train_df.columns.tolist()}")

    # --- 2. Train the Agent ---
    # Make sure the TECH_INDICATOR_LIST in env config matches columns in train_df/eval_df
    # Example: if train_df has 'macd_0', 'rsi_30_0', then TECH_INDICATOR_LIST should be ["macd", "rsi_30"]

    # Verify required columns for the environment are present
    env_check_df = train_df.copy()
    # The env adds momentum and volume_trend itself if not present, but let's assume they are from preprocessing
    # required_env_cols = []
    # for stock_id in range(NUM_STOCKS):
    #     required_env_cols.extend([f'momentum_{stock_id}', f'volume_trend_{stock_id}'])
    # missing_cols = [col for col in required_env_cols if col not in env_check_df.columns]
    # if missing_cols:
    #     print(f"Warning: The following columns required by the environment's _preprocess_data are missing from the dummy data and will be created by the env: {missing_cols}")
    # else:
    #     print("All expected synthetic columns (momentum, volume_trend) are present in dummy data.")


    trained_model = train_agent(train_df, eval_df)

    # --- 3. (Optional) Evaluate or Run the Agent ---
    print("Placeholder for further evaluation or backtesting with the trained model.")
    # Example:
    # test_env = create_env(eval_df, training=False) # Use a separate test dataset
    # obs, _ = test_env.reset()
    # for _ in range(len(eval_df) -1): # run for the length of the test data
    #     action, _states = trained_model.predict(obs, deterministic=True)
    #     obs, rewards, terminated, truncated, info = test_env.step(action)
    #     if terminated or truncated:
    #         print("Episode finished.")
    #         # print(info) # info contains portfolio details
    #         break
    # test_env.close()

    print("Main script finished.")
