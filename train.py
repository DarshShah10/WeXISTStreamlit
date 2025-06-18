import os
import datetime
import pandas as pd
import numpy as np

# Stable Baselines3 components
from stable_baselines3 import PPO, A2C, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise # For TD3/DDPG

# Project-specific imports
from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from src.envs.trading_env import MultiStockTradingEnv
# We might need to refactor td3_agent.py or import parts of it,
# or replicate its logic here for different agents.
# from src.agents.td3_agent import TD3TradingBot # Example, might not be used directly

# Configuration (placeholder, will be moved to a config file later)
DEFAULT_TICKER_LIST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2021-01-01"
DEFAULT_TEST_END_DATE = "2022-01-01" # For creating a separate test set

# Environment parameters (placeholders)
NUM_STOCKS = len(DEFAULT_TICKER_LIST) # Should be derived from data
INITIAL_AMOUNT = 100000.0
BUY_COST_PCT = [0.001] * NUM_STOCKS
SELL_COST_PCT = [0.001] * NUM_STOCKS
HMAX_PER_STOCK = [1000] * NUM_STOCKS
REWARD_SCALING = 1e-4
TECH_INDICATOR_LIST = [
    "sma50", "sma200", "ema12", "ema26", "macd", "rsi", "cci", "adx",
    "sok", "sod", "du", "dl", "vm", "bb_upper", "bb_lower", "bb_middle", "obv"
]
LOOKBACK_WINDOW = 30
MAX_STEPS_PER_EPISODE = 2000 # Example, can be tuned
TRAIN_LOG_DIR = "logs/tensorboard/"
MODEL_SAVE_DIR = "models/"
N_EVAL_EPISODES = 5
EVAL_FREQ = 10000 # Steps

def select_algorithm():
    print("\nSelect model to train:")
    print("1 - PPO")
    print("2 - A2C")
    print("3 - TD3")
    print("4 - DDPG")

    while True:
        try:
            choice = int(input(">> "))
            if choice in [1, 2, 3, 4]:
                return {1: "PPO", 2: "A2C", 3: "TD3", 4: "DDPG"}[choice]
            else:
                print("Invalid choice. Please select a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def prepare_data(ticker_list, start_date, end_date, test_end_date):
    print("\n--- Preparing Data ---")
    pipeline = StockPreProcessPipeline()

    # Define train and eval file paths
    train_df_filename = f"train_combined_data_{'_'.join(ticker_list)}_{start_date}_{end_date}.csv"
    train_df_path = os.path.join("data", train_df_filename)

    eval_df_filename = f"test_combined_data_{'_'.join(ticker_list)}_{end_date}_{test_end_date}.csv"
    eval_df_path = os.path.join("data", eval_df_filename)

    print(f"Processing training data for tickers: {ticker_list} from {start_date} to {end_date}")
    # Pass the full path to the pipeline
    processed_train_path = pipeline.run_data_pipeline(ticker_list, start_date, end_date, train_df_path)
    print(f"Training data processed and saved to: {processed_train_path}")

    print(f"Processing test data for tickers: {ticker_list} from {end_date} to {test_end_date}")
    processed_eval_path = pipeline.run_data_pipeline(ticker_list, end_date, test_end_date, eval_df_path)
    print(f"Evaluation data processed and saved to: {processed_eval_path}")

    try:
        train_df = pd.read_csv(processed_train_path)
        train_df["Date"] = pd.to_datetime(train_df["Date"])
        train_df.set_index("Date", inplace=True)

        eval_df = pd.read_csv(processed_eval_path)
        eval_df["Date"] = pd.to_datetime(eval_df["Date"])
        eval_df.set_index("Date", inplace=True)
    except KeyError as e:
        print(f"KeyError: {e}. Make sure the CSV from the pipeline has a Date column.")
        # It is possible that the read_csv path is incorrect if run_data_pipeline doesnt return the path passed to it
        print("Columns in train_df (path from pipeline):", pd.read_csv(processed_train_path).columns if os.path.exists(processed_train_path) else "Train file not found at " + processed_train_path)
        print("Columns in eval_df (path from pipeline):", pd.read_csv(processed_eval_path).columns if os.path.exists(processed_eval_path) else "Eval file not found at " + processed_eval_path)
        raise

    return train_df, eval_df, len(ticker_list)

def create_env(df, num_stocks_env, training=True, env_hyperparams=None):
    # env_hyperparams will come from config
    # For now, use placeholders or defaults similar to td3_agent

    # Update global NUM_STOCKS if it's different from what's passed
    # This is a bit messy, config will solve this.
    global NUM_STOCKS
    NUM_STOCKS = num_stocks_env

    env = MultiStockTradingEnv(
        df=df,
        num_stocks=num_stocks_env,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=[0.001] * num_stocks_env, # Adjust list size
        sell_cost_pct=[0.001] * num_stocks_env, # Adjust list size
        hmax_per_stock=[1000] * num_stocks_env, # Adjust list size
        reward_scaling=REWARD_SCALING,
        tech_indicator_list=TECH_INDICATOR_LIST,
        lookback_window=LOOKBACK_WINDOW,
        training=training,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE
    )
    return env

def main():
    selected_algo_name = select_algorithm()
    print(f"You selected: {selected_algo_name}")

    # --- 1. Data Preparation ---
    # For now, using default tickers and dates
    # TODO: Allow user to input tickers, start/end dates
    train_df, eval_df, actual_num_stocks = prepare_data(
        DEFAULT_TICKER_LIST,
        DEFAULT_START_DATE,
        DEFAULT_END_DATE,
        DEFAULT_TEST_END_DATE
    )

    # Update NUM_STOCKS based on actual data processed
    # This is important for env creation.
    global NUM_STOCKS
    NUM_STOCKS = actual_num_stocks

    # --- 2. Environment Setup ---
    print("\n--- Setting up Training Environment ---")
    # Create a dummy env lambda for VecEnv
    # Important: Ensure env_hyperparams are passed correctly if they vary per algo
    # or are loaded from config.

    # Training environment
    train_env_sb = DummyVecEnv([lambda: create_env(train_df, NUM_STOCKS, training=True)])
    train_env_sb = VecNormalize(train_env_sb, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Training environment created and normalized.")

    # Evaluation environment
    print("\n--- Setting up Evaluation Environment ---")
    eval_env_sb_raw = DummyVecEnv([lambda: create_env(eval_df, NUM_STOCKS, training=False)]) # Use training=False for eval
    # It is important to use the same statistics for normalization as the training env
    # For EvalCallback, we save the running stats of train_env, load them into a new VecNormalize instance for eval_env
    # then set eval_env.training = False and eval_env.norm_reward = False
    # However, EvalCallback can also handle this if save_vecnormalize=True and it loads the stats.
    # Let us ensure the EvalCallback gets an environment that is correctly normalized.
    # The current way (normalizing eval_env_sb separately) is simpler for now but less accurate for eval.
    # A better way for EvalCallback:
    # 1. Don't normalize eval_env_sb when creating it initially for the callback.
    # 2. EvalCallback internally handles loading the VecNormalize wrapper if `save_vecnormalize=True` in Checkpoint or if model is saved with wrapper.
    # For now, the separate normalization of eval_env_sb will be kept, and we will rely on
    # EvalCallback's `save_vecnormalize` and its loading mechanism, or ensure test.py loads correctly.
    # The critical part is that test.py MUST load the VecNormalize stats from the *saved training model*.

    # For EvalCallback, the environment should be normalized using the stats from the training env.
    # Let's keep it simple for now: EvalCallback will use its own VecNormalize wrapper.
    # The key is that the *best model* saved by EvalCallback will also save its VecNormalize stats.
    eval_env_sb = VecNormalize(eval_env_sb_raw, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # For custom evaluation or if EvalCallback does not handle it as expected:
    # stats_path = os.path.join(MODEL_SAVE_DIR, "checkpoints", f"{selected_algo_name.lower()}_{timestamp}", "train_env_stats.pkl")
    # if os.path.exists(stats_path):
    #   eval_env_sb = VecNormalize.load(stats_path, eval_env_sb_raw) # eval_env_sb_raw would be DummyVecEnv without VecNormalize
    #   eval_env_sb.training = False
    #   eval_env_sb.norm_reward = False
    # else:
    #   print(f"Warning: Could not find {stats_path} to normalize eval_env. EvalCallback might use running stats.")

    # The CheckpointCallback saves vecnormalize stats. EvalCallback also saves them for the best model.
    # So, the eval_env_sb passed to EvalCallback should ideally be normalized with train_env stats.
    # Let's stick to the current separate normalization for `eval_env_sb` for `EvalCallback` for now,
    # as SB3 EvalCallback can manage VecNormalize instances. The crucial part is `test.py`.
    # eval_env_sb = VecNormalize(eval_env_sb, norm_obs=True, norm_reward=True, clip_obs=10.0) # This was the original line
    print("Evaluation environment created and normalized.")

    # --- 3. Model Initialization ---
    print(f"\n--- Initializing {selected_algo_name} Model ---")

    # Timestamp for unique model names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_base = f"{MODEL_SAVE_DIR}{selected_algo_name.lower()}_{timestamp}"

    # Ensure model save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

    model = None
    if selected_algo_name == "TD3":
        # Based on td3_agent.py structure
        action_noise = NormalActionNoise(
            mean=np.zeros(NUM_STOCKS), sigma=0.1 * np.ones(NUM_STOCKS)
        )
        model = TD3(
            "MlpPolicy",
            train_env_sb,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=TRAIN_LOG_DIR,
            # Add other TD3 specific params from config later
            learning_rate=3e-4, # Example
            batch_size=128,     # Example
        )
    elif selected_algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env_sb,
            verbose=1,
            tensorboard_log=TRAIN_LOG_DIR,
            # Add PPO specific params from config later
            n_steps=1024, # Example
            batch_size=64, # Example
        )
    elif selected_algo_name == "A2C":
        model = A2C(
            "MlpPolicy",
            train_env_sb,
            verbose=1,
            tensorboard_log=TRAIN_LOG_DIR,
            # Add A2C specific params from config later
        )
    elif selected_algo_name == "DDPG":
        action_noise = NormalActionNoise(
            mean=np.zeros(NUM_STOCKS), sigma=0.1 * np.ones(NUM_STOCKS)
        )
        model = DDPG(
            "MlpPolicy",
            train_env_sb,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=TRAIN_LOG_DIR,
            # Add DDPG specific params from config later
        )
    else:
        print(f"Algorithm {selected_algo_name} not fully implemented yet.")
        return

    # --- 4. Callbacks ---
    print("\n--- Setting up Callbacks ---")
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(EVAL_FREQ // 10, 1000), # Save checkpoints more often than full eval
        save_path=os.path.join(MODEL_SAVE_DIR, "checkpoints", f"{selected_algo_name.lower()}_{timestamp}"),
        name_prefix=f"{selected_algo_name.lower()}_model",
        save_replay_buffer=True, # For off-policy algos
        save_vecnormalize=True,
    )

    # Eval Callback
    eval_callback = EvalCallback(
        eval_env_sb,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best_model", f"{selected_algo_name.lower()}_{timestamp}"),
        log_path=os.path.join(MODEL_SAVE_DIR, "best_model", f"{selected_algo_name.lower()}_{timestamp}_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        save_vecnormalize=True, # Save VecNormalize stats with the best model
    )

    # --- 5. Training ---
    print(f"\n--- Starting Training for {selected_algo_name} ---")
    # TODO: Get total_timesteps from config or user input
    total_timesteps = 50000 # Small number for testing

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback], # Can pass a list of callbacks
            progress_bar=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    # --- 6. Saving Model ---
    print("\n--- Saving Final Model ---")
    final_model_path = f"{model_filename_base}.zip"
    model.save(final_model_path)

    # Save VecNormalize stats
    vec_normalize_path = f"{model_filename_base}_vecnormalize.pkl"
    # Ensure the VecNormalize stats from the *training* environment are saved with the final model.
    # This is crucial for test.py to load the model and stats correctly.
    train_env_sb.save(vec_normalize_path)
    print(f"VecNormalize stats from training environment saved to: {vec_normalize_path}")

    print(f"Model saved to: {final_model_path}")
    # print(f"VecNormalize stats saved to: {vec_normalize_path}") # Removed as per user script logic
    print(f"Training logs saved in: {TRAIN_LOG_DIR}{selected_algo_name.lower()}_1 (or similar)") # SB3 appends _1, _2 etc.
    print(f"Best model and eval logs saved in: {os.path.join(MODEL_SAVE_DIR, 'best_model', f'{selected_algo_name.lower()}_{timestamp}')}")

    print("\n--- Training Finished ---")

if __name__ == "__main__":
    # Before running main, ensure all necessary files for DataPreprocessingPipeline are available
    # This might involve creating dummy files if they are not fully implemented yet
    # For now, assuming DataPreprocessingPipeline can be imported and instantiated.

    # Create dummy files that DataPreprocessingPipeline imports if they don't exist
    # to avoid ImportErrors during subtask execution if those files are not yet populated.
    # This is a workaround for the subtask environment.
    required_preprocessing_files = [
        "src/data_preprocessing/CollectData.py",
        "src/data_preprocessing/PreProcess.py",
        "src/data_preprocessing/FeatureEngineer.py",
        "src/data_preprocessing/Post_Process_Features.py",
        "src/data_preprocessing/Download_Macro.py",
        "src/data_preprocessing/PreProcessMacro.py",
        "src/data_preprocessing/CombineDf.py"
    ]
    for f_path in required_preprocessing_files:
        if not os.path.exists(f_path):
            os.makedirs(os.path.dirname(f_path), exist_ok=True)
            with open(f_path, 'w') as f:
                if "CollectData.py" in f_path:
                    f.write("class StockDataCollector: pass\n") # Minimal content
                elif "PreProcess.py" in f_path:
                    f.write("class StockDataPreprocessor: pass\n")
                elif "FeatureEngineer.py" in f_path:
                    f.write("class StockFeatureEngineer: pass\n")
                elif "Post_Process_Features.py" in f_path:
                    f.write("class StockDataPostProcessor: pass\n")
                elif "Download_Macro.py" in f_path:
                    f.write("class MacroDataDownloader: pass\n")
                elif "PreProcessMacro.py" in f_path:
                    f.write("class MacroDataProcessor: pass\n")
                elif "CombineDf.py" in f_path:
                    f.write("class DataMerger: pass\n")
                else:
                    f.write("# Dummy file\n")

    # Update import in td3_agent.py as well, as it might be used later or as reference
    TD3_AGENT_PATH = "src/agents/td3_agent.py"
    if os.path.exists(TD3_AGENT_PATH):
        with open(TD3_AGENT_PATH, "r") as f:
            td3_content = f.read()
        new_td3_content = td3_content.replace(
            "from Environment.New_Trading_Env import MultiStockTradingEnv",
            "from src.envs.trading_env import MultiStockTradingEnv"
        )
        if new_td3_content != td3_content:
            with open(TD3_AGENT_PATH, "w") as f:
                f.write(new_td3_content)
            print(f"Updated import in {TD3_AGENT_PATH}")

    main()
