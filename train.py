import os
import datetime
import pandas as pd
import numpy as np

# Stable Baselines3 components
from stable_baselines3 import PPO, A2C, TD3, DDPG
from sb3_contrib import RecurrentPPO # Added for RecurrentPPO
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
    print("5 - RecurrentPPO")

    while True:
        try:
            choice = int(input(">> "))
            if choice in [1, 2, 3, 4, 5]:
                return {1: "PPO", 2: "A2C", 3: "TD3", 4: "DDPG", 5: "RecurrentPPO"}[choice]
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
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

def run_training_programmatically(
    selected_algo_name_param: str,
    ticker_list_param: list[str],
    start_date_param: str,
    end_date_param: str,
    test_end_date_param: str,
    total_timesteps_param: int,
    model_tag_param: str = ""
):
    """
    Runs the training process programmatically.
    """
    print(f"Starting programmatic training for {selected_algo_name_param} with tickers: {ticker_list_param}")
    print(f"Data range: {start_date_param} to {end_date_param}, Test end date: {test_end_date_param}")
    print(f"Total timesteps: {total_timesteps_param}, Model tag: '{model_tag_param}'")

    # --- 1. Data Preparation ---
    train_df, eval_df, actual_num_stocks = prepare_data(
        ticker_list_param,
        start_date_param,
        end_date_param,
        test_end_date_param
    )

    # Update NUM_STOCKS based on actual data processed
    # This is important for env creation.
    global NUM_STOCKS
    NUM_STOCKS = actual_num_stocks

    # --- 2. Environment Setup ---
    print("\n--- Setting up Training Environment ---")
    train_env_sb = DummyVecEnv([lambda: create_env(train_df, NUM_STOCKS, training=True)])
    train_env_sb = VecNormalize(train_env_sb, norm_obs=True, norm_reward=True, clip_obs=10.0)
    print("Training environment created and normalized.")

    print("\n--- Setting up Evaluation Environment ---")
    eval_env_sb_raw = DummyVecEnv([lambda: create_env(eval_df, NUM_STOCKS, training=False)])
    eval_env_sb = VecNormalize(eval_env_sb_raw, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Note: For robust evaluation, eval_env_sb should ideally be normalized with stats from train_env_sb.
    # EvalCallback handles saving/loading vecnormalize stats for the best model.
    print("Evaluation environment created and normalized.")

    # --- 3. Model Initialization ---
    print(f"\n--- Initializing {selected_algo_name_param} Model ---")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_prefix = selected_algo_name_param.lower()
    if model_tag_param:
        model_name_prefix = f"{model_name_prefix}_{model_tag_param}"

    model_filename_base = f"{MODEL_SAVE_DIR}{model_name_prefix}_{timestamp}"
    checkpoint_save_path = os.path.join(MODEL_SAVE_DIR, "checkpoints", f"{model_name_prefix}_{timestamp}")
    best_model_save_dir = os.path.join(MODEL_SAVE_DIR, "best_model", f"{model_name_prefix}_{timestamp}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True) # Tensorboard log dir
    os.makedirs(checkpoint_save_path, exist_ok=True)
    os.makedirs(best_model_save_dir, exist_ok=True)


    model = None
    if selected_algo_name_param == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(NUM_STOCKS), sigma=0.1 * np.ones(NUM_STOCKS)
        )
        model = TD3(
            "MlpPolicy", train_env_sb, action_noise=action_noise, verbose=1,
            tensorboard_log=TRAIN_LOG_DIR, learning_rate=3e-4, batch_size=128,
            gamma=0.99, tau=0.005, policy_delay=2, policy_kwargs=dict(net_arch=[400, 300])
        )
    elif selected_algo_name_param == "PPO":
        model = PPO(
            "MlpPolicy", train_env_sb, verbose=1, tensorboard_log=TRAIN_LOG_DIR,
            n_steps=1024, batch_size=64
        )
    elif selected_algo_name_param == "A2C":
        model = A2C(
            "MlpPolicy", train_env_sb, verbose=1, tensorboard_log=TRAIN_LOG_DIR
        )
    elif selected_algo_name_param == "DDPG":
        action_noise = NormalActionNoise(
            mean=np.zeros(NUM_STOCKS), sigma=0.1 * np.ones(NUM_STOCKS)
        )
        model = DDPG(
            "MlpPolicy", train_env_sb, action_noise=action_noise, verbose=1,
            tensorboard_log=TRAIN_LOG_DIR
        )
    elif selected_algo_name_param == "RecurrentPPO":
        model = RecurrentPPO(
            "LstmPolicy", train_env_sb, verbose=1, tensorboard_log=TRAIN_LOG_DIR,
            learning_rate=3e-4, batch_size=128, n_steps=2048, gamma=0.99,
            gae_lambda=0.95, ent_coef=0.01, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[400, 300])
        )
    else:
        print(f"Algorithm {selected_algo_name_param} not fully implemented yet.")
        return None # Or raise error

    # --- 4. Callbacks ---
    print("\n--- Setting up Callbacks ---")
    checkpoint_callback = CheckpointCallback(
        save_freq=max(EVAL_FREQ // 10, 1000),
        save_path=checkpoint_save_path, # Use updated path
        name_prefix=f"{selected_algo_name_param.lower()}_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env_sb,
        best_model_save_path=best_model_save_dir, # Use updated path
        log_path=os.path.join(best_model_save_dir, "logs"), # Logs within the best_model_save_dir
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        save_vecnormalize=True,
    )

    # --- 5. Training ---
    print(f"\n--- Starting Training for {selected_algo_name_param} ---")
    try:
        model.learn(
            total_timesteps=total_timesteps_param,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True # Consider making this a parameter if running in non-interactive mode
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise # Re-raise the exception to be caught by the caller if necessary

    # --- 6. Saving Model ---
    print("\n--- Saving Final Model ---")
    final_model_path = f"{model_filename_base}.zip"
    model.save(final_model_path)

    vec_normalize_path = f"{model_filename_base}_vecnormalize.pkl"
    train_env_sb.save(vec_normalize_path)
    print(f"VecNormalize stats from training environment saved to: {vec_normalize_path}")

    print(f"Model saved to: {final_model_path}")
    # Tensorboard logs are typically under TRAIN_LOG_DIR/ALGO_1, ALGO_2 etc.
    print(f"Training logs (TensorBoard) likely in: {TRAIN_LOG_DIR}")
    print(f"Best model and eval logs saved in: {best_model_save_dir}")
    print("\n--- Programmatic Training Finished ---")
    return final_model_path

def main():
    selected_algo_name = select_algorithm()
    print(f"You selected: {selected_algo_name}")

    # Example of calling the programmatic function
    # In a real CLI, these would come from args
    total_timesteps_cli = 50000 # Default for CLI, can be an arg
    model_tag_cli = "cli_run"   # Example tag for CLI runs

    print(f"\n--- Starting training process via main() wrapper ---")
    try:
        saved_model_path = run_training_programmatically(
            selected_algo_name_param=selected_algo_name,
            ticker_list_param=DEFAULT_TICKER_LIST,
            start_date_param=DEFAULT_START_DATE,
            end_date_param=DEFAULT_END_DATE,
            test_end_date_param=DEFAULT_TEST_END_DATE,
            total_timesteps_param=total_timesteps_cli,
            model_tag_param=model_tag_cli
        )
        if saved_model_path:
            print(f"Training initiated by main() completed. Model saved at: {saved_model_path}")
        else:
            print(f"Training initiated by main() failed or algorithm was not found.")
    except Exception as e:
        print(f"An error occurred during training initiated by main(): {e}")
        # Optionally, re-raise or handle more gracefully
        # raise

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
