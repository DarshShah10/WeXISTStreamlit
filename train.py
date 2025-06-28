import os
import datetime
import pandas as pd
import numpy as np
import logging # For logging config loading issues

# Stable Baselines3 components
from stable_baselines3 import PPO, A2C, TD3, DDPG
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Project-specific imports
from src.utils import load_config # Import the new config loader
from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from src.envs.trading_env import MultiStockTradingEnv

# --- Global Configuration ---
# Load configuration at the start. Paths will be absolute.
try:
    CONFIG = load_config() # Default path: "config/shared_config.yaml"
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    # Fallback to some critical defaults if config fails, or exit
    # For this refactor, we'll assume config loads. Production code might need robust fallbacks.
    CONFIG = {} # Initialize to empty dict to avoid KeyErrors later, though script might fail.
    # Consider raising the exception if config is critical: raise

# Remove old global constants, they will come from CONFIG
# DEFAULT_TICKER_LIST, DEFAULT_START_DATE, etc. are gone.
# NUM_STOCKS, INITIAL_AMOUNT, etc. are gone.

def select_algorithm():
    print("\nSelect model to train:")
    print("1 - PPO")
    print("2 - A2C")
    print("3 - TD3")
    print("4 - DDPG")
    print("5 - RecurrentPPO (from sb3_contrib)") # Clarify source

    while True:
        try:
            choice = int(input(">> "))
            # Ensure choices align with how model config is stored (e.g., 'ppo', 'td3')
            algo_map = {1: "PPO", 2: "A2C", 3: "TD3", 4: "DDPG", 5: "RecurrentPPO"}
            if choice in algo_map:
                return algo_map[choice]
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(algo_map)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def prepare_data(config, ticker_list, start_date, end_date, test_end_date):
    print("\n--- Preparing Data ---")
    pipeline = StockPreProcessPipeline(config) # Pass the loaded config

    # Construct output paths for processed data using config
    # Ensure final_combined_data_dir exists
    final_data_dir = config.get('final_combined_data_dir', os.path.join(config.get('project_root', '.'), 'data', 'final_combined_data'))
    os.makedirs(final_data_dir, exist_ok=True)

    # Standardized filenames for train/eval sets, or make these configurable too
    train_df_filename = f"train_data_{'_'.join(ticker_list)}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
    train_df_path = os.path.join(final_data_dir, train_df_filename)

    eval_df_filename = f"eval_data_{'_'.join(ticker_list)}_{end_date.replace('-', '')}_{test_end_date.replace('-', '')}.csv"
    eval_df_path = os.path.join(final_data_dir, eval_df_filename)

    print(f"Processing training data for tickers: {ticker_list} from {start_date} to {end_date}")
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

    # Determine num_actual_stocks based on the 'Ticker' column if available, otherwise fallback to ticker_list length.
    # This is more robust if some tickers yield no data.
    num_actual_stocks = len(train_df.Ticker.unique()) if 'Ticker' in train_df.columns else len(ticker_list)
    return train_df, eval_df, num_actual_stocks


def create_env(df, num_stocks_env, config, training=True):
    env_params = config # Directly use the main config for env parameters

    # Ensure buy/sell cost and hmax lists match num_stocks_env
    buy_cost_pct_config = env_params.get('buy_cost_pct', 0.001)
    if isinstance(buy_cost_pct_config, float):
        buy_cost_pct = [buy_cost_pct_config] * num_stocks_env
    elif isinstance(buy_cost_pct_config, list) and len(buy_cost_pct_config) == num_stocks_env:
        buy_cost_pct = buy_cost_pct_config
    else: # Fallback: use first element if list, or default, for all stocks
        default_buy_cost = buy_cost_pct_config[0] if isinstance(buy_cost_pct_config, list) and buy_cost_pct_config else 0.001
        buy_cost_pct = [default_buy_cost] * num_stocks_env
        logging.warning(f"Buy cost mismatch or format error. Using {default_buy_cost} for all stocks.")

    sell_cost_pct_config = env_params.get('sell_cost_pct', 0.001)
    if isinstance(sell_cost_pct_config, float):
        sell_cost_pct = [sell_cost_pct_config] * num_stocks_env
    elif isinstance(sell_cost_pct_config, list) and len(sell_cost_pct_config) == num_stocks_env:
        sell_cost_pct = sell_cost_pct_config
    else: # Fallback
        default_sell_cost = sell_cost_pct_config[0] if isinstance(sell_cost_pct_config, list) and sell_cost_pct_config else 0.001
        sell_cost_pct = [default_sell_cost] * num_stocks_env
        logging.warning(f"Sell cost mismatch or format error. Using {default_sell_cost} for all stocks.")

    hmax_per_stock_config = env_params.get('hmax_per_stock', 1000)
    if isinstance(hmax_per_stock_config, (int, float)):
        hmax_per_stock = [hmax_per_stock_config] * num_stocks_env
    elif isinstance(hmax_per_stock_config, list) and len(hmax_per_stock_config) == num_stocks_env:
        hmax_per_stock = hmax_per_stock_config
    else: # Fallback
        default_hmax = hmax_per_stock_config[0] if isinstance(hmax_per_stock_config, list) and hmax_per_stock_config else 1000
        hmax_per_stock = [default_hmax] * num_stocks_env
        logging.warning(f"HMAX per stock mismatch or format error. Using {default_hmax} for all stocks.")

    env = MultiStockTradingEnv(
        df=df,
        num_stocks=num_stocks_env,
        tech_indicator_list=config.get('tech_indicator_list', []),
        initial_amount=env_params.get('initial_amount', 100000.0),
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        hmax_per_stock=hmax_per_stock,
        reward_scaling=env_params.get('reward_scaling', 1e-4),
        lookback_window=env_params.get('lookback_window', 30),
        max_steps_per_episode=env_params.get('max_steps_per_episode', 2000),
        reward_params=env_params.get('reward_params', {}), # Pass reward sub-config
        training=training,
        # stop_loss_threshold=env_params.get('stop_loss_threshold', 0.15), # Add if env supports
        # risk_penalty=env_params.get('risk_penalty', 0.0005) # Add if env supports
    )
    return env

def _get_action_noise(config_noise_params, num_stocks):
    """Helper to create action noise object from config."""
    if not config_noise_params or not isinstance(config_noise_params, dict):
        logging.info("No action noise parameters found or invalid format. No action noise will be used.")
        return None

    noise_type = config_noise_params.get("type", "NormalActionNoise").lower()
    mean_param = config_noise_params.get("mean", 0.0)
    sigma_param = config_noise_params.get("sigma", 0.1)

    # Ensure mean and sigma are correctly sized for num_stocks
    # If single value, create a list/array of that value for all stocks
    mean_val = np.full(num_stocks, mean_param) if isinstance(mean_param, (float, int)) else np.array(mean_param)
    sigma_val = np.full(num_stocks, sigma_param) if isinstance(sigma_param, (float, int)) else np.array(sigma_param)

    if mean_val.shape[0] != num_stocks :
        logging.warning(f"Action noise mean size mismatch. Expected {num_stocks}, got {mean_val.shape[0]}. Using default mean 0.")
        mean_val = np.zeros(num_stocks)
    if sigma_val.shape[0] != num_stocks:
        logging.warning(f"Action noise sigma size mismatch. Expected {num_stocks}, got {sigma_val.shape[0]}. Using default sigma 0.1.")
        sigma_val = np.full(num_stocks, 0.1)


    if noise_type == "normalactionnoise":
        logging.info(f"Using NormalActionNoise with mean {mean_val.tolist()} and sigma {sigma_val.tolist()}")
        return NormalActionNoise(mean=mean_val, sigma=sigma_val)
    elif noise_type == "ornsteinuhlenbeckactionnoise":
        # OU noise might have other params like theta, dt. SB3's OU takes sigma.
        logging.info(f"Using OrnsteinUhlenbeckActionNoise with mean {mean_val.tolist()} and sigma {sigma_val.tolist()}")
        return OrnsteinUhlenbeckActionNoise(mean=mean_val, sigma=sigma_val)
    else:
        logging.warning(f"Unsupported action noise type: {noise_type}. No action noise will be used.")
        return None

def run_training_programmatically(
    config, # Pass the full config object
    selected_algo_name: str,
    ticker_list: list[str],
    start_date: str,
    end_date: str,
    test_end_date: str,
    total_timesteps: int,
    model_tag: str = ""
):
    print(f"Starting programmatic training for {selected_algo_name} with tickers: {ticker_list}")
    print(f"Data range: {start_date} to {end_date}, Test end date: {test_end_date}")
    print(f"Total timesteps: {total_timesteps}, Model tag: '{model_tag}'")

    # --- 1. Data Preparation ---
    # Pass config to prepare_data
    train_df, eval_df, actual_num_stocks = prepare_data(
        config, ticker_list, start_date, end_date, test_end_date
    )
    # actual_num_stocks is now determined by prepare_data based on actual data
    # This is crucial for env and model setup.
    logging.info(f"Actual number of stocks successfully processed: {actual_num_stocks}")
    if actual_num_stocks == 0:
        logging.error("No stock data processed. Aborting training.")
        return None


    # --- 2. Environment Setup ---
    print("\n--- Setting up Training Environment ---")
    # Pass the main config to create_env, it will pick relevant parts
    train_env_sb = DummyVecEnv([lambda: create_env(train_df, actual_num_stocks, config, training=True)])
    train_env_sb = VecNormalize(train_env_sb, norm_obs=True, norm_reward=True, clip_obs=10.0) # TODO: clip_obs from config?
    print("Training environment created and normalized.")

    print("\n--- Setting up Evaluation Environment ---")
    eval_env_sb_raw = DummyVecEnv([lambda: create_env(eval_df, actual_num_stocks, config, training=False)])
    # For evaluation, use the stats from the training env for normalization
    # Must save and load VecNormalize stats correctly for this. EvalCallback handles saving for best model.
    # Here, we manually ensure the eval_env for EvalCallback is normalized using train_env stats.
    eval_stats_path = os.path.join(config.get('model_dir', 'models'), f"temp_eval_vec_normalize_stats_for_{selected_algo_name}.pkl")
    train_env_sb.save_running_average(os.path.dirname(eval_stats_path)) # Save current stats
    eval_env_sb = VecNormalize.load(eval_stats_path, eval_env_sb_raw)
    eval_env_sb.training = False # Important: set to False after loading
    eval_env_sb.norm_reward = False # Usually, don't normalize rewards for evaluation
    os.remove(eval_stats_path) # Clean up temporary file
    print("Evaluation environment created and normalized using training stats.")


    # --- 3. Model Initialization ---
    print(f"\n--- Initializing {selected_algo_name} Model ---")

    algo_key = selected_algo_name.lower() # e.g., "ppo", "td3"
    model_specific_params = config.get(algo_key, {}) # Get params for the specific algo e.g. config['ppo']
    default_policy = config.get('default_policy', 'MlpPolicy')
    policy = model_specific_params.get('policy', default_policy) # Algo can override default policy

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_with_tag = algo_key
    if model_tag:
        model_name_with_tag = f"{model_name_with_tag}_{model_tag}"

    # Paths from config (config_loader should have made them absolute)
    # These are base directories for storing models and logs for this type of run
    tensorboard_log_dir = config.get('tensorboard_log_dir') # e.g., /app/logs/tensorboard_logs
    checkpoint_base_dir = config.get('checkpoint_dir')     # e.g., /app/models/checkpoints
    best_model_base_dir = config.get('best_model_dir')     # e.g., /app/models/best_model
    # This is the general directory where all models of this type/tag/timestamp will be stored
    run_specific_model_artifacts_dir = os.path.join(config.get('model_dir'), f"{model_name_with_tag}_{timestamp}")


    # Specific paths for this run's artifacts
    # Final model will be saved in run_specific_model_artifacts_dir
    final_model_filename_base = os.path.join(run_specific_model_artifacts_dir, model_name_with_tag)

    # Checkpoints and best models will have their own subdirectories under their respective base_dirs
    checkpoint_run_dir = os.path.join(checkpoint_base_dir, f"{model_name_with_tag}_{timestamp}")
    best_model_run_dir = os.path.join(best_model_base_dir, f"{model_name_with_tag}_{timestamp}")

    os.makedirs(run_specific_model_artifacts_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True) # This is a shared dir for all runs
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    os.makedirs(best_model_run_dir, exist_ok=True)

    model = None
    # Common params for many models, can be overridden by specific algo config
    shared_model_kwargs = {
        "policy": policy,
        "env": train_env_sb,
        "verbose": model_specific_params.get('verbose', 1), # Use verbose from algo_config or default
        "tensorboard_log": tensorboard_log_dir, # Shared TB log dir
        "learning_rate": model_specific_params.get('learning_rate', config.get('ppo',{}).get('learning_rate', 3e-4)), # Example of deeper default
        "policy_kwargs": model_specific_params.get('policy_kwargs', None)
    }

    if isinstance(shared_model_kwargs['policy_kwargs'], str):
        try:
            import ast
            shared_model_kwargs['policy_kwargs'] = ast.literal_eval(shared_model_kwargs['policy_kwargs'])
            logging.info(f"Successfully parsed policy_kwargs: {shared_model_kwargs['policy_kwargs']}")
        except (ValueError, SyntaxError) as e:
            logging.warning(f"Could not parse policy_kwargs string: {shared_model_kwargs['policy_kwargs']}. Error: {e}. Using None.")
            shared_model_kwargs['policy_kwargs'] = None


    if selected_algo_name == "TD3":
        td3_params = model_specific_params
        action_noise_params = td3_params.get('action_noise')
        action_noise = _get_action_noise(action_noise_params, actual_num_stocks)

        model = TD3(
            **shared_model_kwargs,
            action_noise=action_noise,
            batch_size=td3_params.get('batch_size', 128),
            buffer_size=td3_params.get('buffer_size', 1_000_000),
            gamma=td3_params.get('gamma', 0.99),
            tau=td3_params.get('tau', 0.005),
            policy_delay=td3_params.get('policy_delay', 2)
        )
    elif selected_algo_name == "PPO":
        ppo_params = model_specific_params
        model = PPO(
            **shared_model_kwargs,
            n_steps=ppo_params.get('n_steps', 1024),
            batch_size=ppo_params.get('batch_size', 64),
            ent_coef=ppo_params.get('ent_coef', 0.0),
            gae_lambda=ppo_params.get('gae_lambda', 0.95)
        )
    elif selected_algo_name == "A2C":
        a2c_params = model_specific_params
        model = A2C(
            **shared_model_kwargs,
            n_steps=a2c_params.get('n_steps', 5),
            gae_lambda=a2c_params.get('gae_lambda', 0.9),
            ent_coef=a2c_params.get('ent_coef', 0.0)
        )
    elif selected_algo_name == "DDPG":
        ddpg_params = model_specific_params
        action_noise_params = ddpg_params.get('action_noise')
        action_noise = _get_action_noise(action_noise_params, actual_num_stocks)
        model = DDPG(
            **shared_model_kwargs,
            action_noise=action_noise,
            batch_size=ddpg_params.get('batch_size', 128),
            buffer_size=ddpg_params.get('buffer_size', 1_000_000),
            gamma=ddpg_params.get('gamma', 0.99),
            tau=ddpg_params.get('tau', 0.005)
        )
    elif selected_algo_name == "RecurrentPPO":
        rppo_params = model_specific_params
        if shared_model_kwargs['policy'] not in ["MlpLstmPolicy", "CnnLstmPolicy", "MultiInputLstmPolicy"]:
             logging.warning(f"RecurrentPPO typically uses a LstmPolicy. Current policy: {shared_model_kwargs['policy']}. Make sure it's compatible or defined in config.")

        model = RecurrentPPO(
            **shared_model_kwargs,
            n_steps=rppo_params.get('n_steps', 2048),
            batch_size=rppo_params.get('batch_size', 128),
            gamma=rppo_params.get('gamma', 0.99),
            gae_lambda=rppo_params.get('gae_lambda', 0.95),
            ent_coef=rppo_params.get('ent_coef', 0.01),
            max_grad_norm=rppo_params.get('max_grad_norm', 0.5)
        )
    else:
        logging.error(f"Algorithm {selected_algo_name} not fully implemented with config.")
        return None

    # --- 4. Callbacks ---
    print("\n--- Setting up Callbacks ---")
    eval_freq = config.get('default_eval_freq', 10000)
    n_eval_episodes = config.get('default_n_eval_episodes', 5)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // 10, 1000),
        save_path=checkpoint_run_dir, # Save checkpoints in their specific run directory
        name_prefix=f"{algo_key}_ckpt", # Simpler prefix
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env_sb,
        best_model_save_path=best_model_run_dir, # Save best model in its specific run directory
        log_path=os.path.join(best_model_run_dir, "eval_logs"), # Logs within the best_model_run_dir
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        save_vecnormalize=True,
    )

    # --- 5. Training ---
    print(f"\n--- Starting Training for {selected_algo_name} ---")
    try:
        model.learn(
            total_timesteps=total_timesteps, # From function arg, ultimately from config
            callback=[checkpoint_callback, eval_callback],
            progress_bar=config.get('training_progress_bar', True) # Make progress bar configurable
        )
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        raise

    # --- 6. Saving Model ---
    print("\n--- Saving Final Model ---")
    # final_model_filename_base is already os.path.join(run_specific_model_artifacts_dir, model_name_with_tag)
    final_model_path = f"{final_model_filename_base}{config.get('model_zip_suffix', '.zip')}"
    model.save(final_model_path)

    # Save VecNormalize stats associated with the final model in the same directory
    vec_normalize_path = f"{final_model_filename_base}{config.get('vecnormalize_suffix', '_vecnormalize.pkl')}"
    train_env_sb.save(vec_normalize_path)
    print(f"VecNormalize stats from training environment saved to: {vec_normalize_path}")

    print(f"Final model saved to: {final_model_path}")
    # Tensorboard logs are in a shared directory, but specific run is identifiable by name (e.g., ppo_1, td3_run_cli_run_1)
    print(f"TensorBoard logs in: {tensorboard_log_dir} (look for a folder like {algo_key}_* or {model_name_with_tag}_*)")
    print(f"Checkpoints saved in: {checkpoint_run_dir}")
    print(f"Best model and eval logs saved in: {best_model_run_dir}")
    print("\n--- Programmatic Training Finished ---")
    return final_model_path


def main():
    if not CONFIG:
        print("Critical error: Configuration could not be loaded. Exiting.")
        logging.critical("Configuration not loaded. Aborting main execution.")
        return

    selected_algo_name = select_algorithm()
    print(f"You selected: {selected_algo_name}")

    # Get parameters from the loaded CONFIG
    ticker_list = CONFIG.get('default_ticker_list', ["RELIANCE.NS", "TCS.NS"])
    start_date = CONFIG.get('default_start_date', "2019-01-01")
    end_date = CONFIG.get('default_end_date', "2022-01-01")
    test_end_date = CONFIG.get('default_test_end_date', "2023-01-01")
    total_timesteps_main = CONFIG.get('default_total_timesteps', 50000)
    # Construct a more descriptive model_tag
    model_tag_main = f"{CONFIG.get('project_name', 'RLTrading')}_{selected_algo_name.lower()}"


    print(f"\n--- Starting training process via main() wrapper with settings from config ---")
    try:
        saved_model_path = run_training_programmatically(
            config=CONFIG,
            selected_algo_name=selected_algo_name, # Renamed for clarity
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            test_end_date=test_end_date,
            total_timesteps=total_timesteps_main,
            model_tag=model_tag_main
        )
        if saved_model_path:
            print(f"Training initiated by main() completed. Model saved at: {saved_model_path}")
        else:
            print(f"Training initiated by main() failed or algorithm was not found/implemented.")
    except Exception as e:
        print(f"An error occurred during training initiated by main(): {e}")
        logging.error(f"An error occurred during training initiated by main(): {e}", exc_info=True)


if __name__ == "__main__":
    # Configure basic logging to see INFO and above
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # Removed dummy file creation and td3_agent.py import fix.
    # Project components should be correctly structured and importable.
    main()
