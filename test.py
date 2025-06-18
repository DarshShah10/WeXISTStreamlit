import os
import glob
import pandas as pd
import numpy as np
import argparse # For potentially passing model path or other params

# Stable Baselines3 components
from stable_baselines3 import PPO, A2C, TD3, DDPG # To load models of different types
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy # Can be used, or manual loop

# Project-specific imports
from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from src.envs.trading_env import MultiStockTradingEnv
from train import ( # Import necessary things from train.py
    create_env,
    prepare_data, # To get test data if needed
    DEFAULT_TICKER_LIST,
    DEFAULT_END_DATE, # This was the end of training data, start of test data
    DEFAULT_TEST_END_DATE, # This was the end of test data
    # MODEL_SAVE_DIR # Already defined below
)

MODEL_SAVE_DIR = "models/" # Defined in train.py as well
DATA_DIR = "data/"

# Placeholder for environment params that create_env needs.
# These should ideally come from a config file shared with train.py
# For now, mirror some critical ones from train.py if not directly imported.
# create_env in train.py uses global variables for these, which is not ideal.
# We will need to pass them or load from config later.
INITIAL_AMOUNT = 100000.0 # from train.py
BUY_COST_PCT = [0.001] # Will be dynamically sized later
SELL_COST_PCT = [0.001] # Will be dynamically sized later
HMAX_PER_STOCK = [1000] # Will be dynamically sized later
REWARD_SCALING = 1e-4 # from train.py
TECH_INDICATOR_LIST = [ # from train.py
    "sma50", "sma200", "ema12", "ema26", "macd", "rsi", "cci", "adx",
    "sok", "sod", "du", "dl", "vm", "bb_upper", "bb_lower", "bb_middle", "obv"
]
LOOKBACK_WINDOW = 30 # from train.py
MAX_STEPS_PER_EPISODE_TEST = 2000 # Can be different from training


def list_models(model_dir):
    print("\nAvailable models:")
    models = []
    # Look for .zip files (actual models)
    zip_files = glob.glob(os.path.join(model_dir, "*.zip"))
    for i, model_path in enumerate(zip_files):
        # Try to find a corresponding _vecnormalize.pkl file
        stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(stats_path):
            models.append({"model_path": model_path, "stats_path": stats_path})
            print(f"{len(models)}) {os.path.basename(model_path)} (with VecNormalize stats)")
        else:
            # Models might be saved without VecNormalize (e.g. if not used in training)
            # or from 'best_model' dir from EvalCallback which might save it differently
            models.append({"model_path": model_path, "stats_path": None})
            print(f"{len(models)}) {os.path.basename(model_path)} (no VecNormalize stats found alongside)")

    # Also check 'best_model' subdirectories, as EvalCallback saves there
    best_model_dirs = glob.glob(os.path.join(model_dir, "best_model", "*"))
    for best_model_parent_dir in best_model_dirs:
        if os.path.isdir(best_model_parent_dir):
            best_model_path = os.path.join(best_model_parent_dir, "best_model.zip")
            best_model_stats_path = os.path.join(best_model_parent_dir, "vecnormalize.pkl")
            if os.path.exists(best_model_path):
                if os.path.exists(best_model_stats_path):
                    models.append({"model_path": best_model_path, "stats_path": best_model_stats_path})
                    print(f"{len(models)}) {os.path.relpath(best_model_path, model_dir)} (Best model with VecNormalize stats)")
                else:
                    models.append({"model_path": best_model_path, "stats_path": None})
                    print(f"{len(models)}) {os.path.relpath(best_model_path, model_dir)} (Best model, no VecNormalize stats found alongside)")

    if not models:
        print("No models found.")
        return None

    while True:
        try:
            choice = int(input("Select model to test (number): "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_model_and_env(selected_model_info, test_df, num_stocks_env):
    model_path = selected_model_info["model_path"]
    stats_path = selected_model_info["stats_path"]

    print(f"\n--- Loading Model: {model_path} ---")

    # Determine model type from filename (PPO, A2C, TD3, DDPG)
    # This is a bit heuristic; config might store this info better.
    model_name_lower = os.path.basename(model_path).lower()
    AlgoClass = None
    if "ppo" in model_name_lower: AlgoClass = PPO
    elif "a2c" in model_name_lower: AlgoClass = A2C
    elif "td3" in model_name_lower: AlgoClass = TD3
    elif "ddpg" in model_name_lower: AlgoClass = DDPG
    else:
        # Fallback for 'best_model.zip' or other names
        # We might need to ask the user or try to infer from structure.
        # For now, let's try PPO as a default or raise error.
        print(f"Could not infer algorithm type from model name: {model_name_lower}. Attempting to load as PPO.")
        print("This might fail if the model is of a different type.")
        AlgoClass = PPO # Default, or could raise an error

    # Create the base environment (not vectorized yet)
    # Use training=False for testing
    # We need to pass the correct num_stocks to create_env
    # The global NUM_STOCKS in train.py is bad. create_env should take it as param.
    # For now, test_df should come from prepare_data which returns num_tickers

    # Hack: Update global env params in train.py before calling create_env
    # This is bad practice and needs to be fixed with config.
    import train
    train.NUM_STOCKS = num_stocks_env
    train.INITIAL_AMOUNT = INITIAL_AMOUNT
    train.BUY_COST_PCT = [0.001] * num_stocks_env
    train.SELL_COST_PCT = [0.001] * num_stocks_env
    train.HMAX_PER_STOCK = [1000] * num_stocks_env
    train.REWARD_SCALING = REWARD_SCALING
    train.TECH_INDICATOR_LIST = TECH_INDICATOR_LIST
    train.LOOKBACK_WINDOW = LOOKBACK_WINDOW
    train.MAX_STEPS_PER_EPISODE = MAX_STEPS_PER_EPISODE_TEST # Use test-specific episode length

    raw_env = create_env(df=test_df, num_stocks_env=num_stocks_env, training=False)

    # Wrap in DummyVecEnv
    vec_env = DummyVecEnv([lambda: raw_env])

    if stats_path and os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from: {stats_path}")
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False  # Important: set to False for testing
        vec_env.norm_reward = False # Usually False for testing
        print("Environment normalized using loaded stats.")
    else:
        print("Warning: No VecNormalize stats found or specified. Using unnormalized environment or environment's own normalization if any.")
        # If the environment itself has internal normalization, that might be used.
        # If model was trained with VecNormalize, this is likely to perform poorly.

    print(f"Loading model from {model_path}...")
    model = AlgoClass.load(model_path, env=vec_env)
    print("Model loaded successfully.")

    return model, vec_env

def run_evaluation(model, env, test_df):
    print("\n--- Running Evaluation ---")
    obs, _ = env.reset()
    # done = False # Not used in this loop structure

    portfolio_values = [INITIAL_AMOUNT] # Start with initial amount
    daily_returns = []

    # Access the underlying Gymnasium env to get detailed info if needed
    # The actual MultiStockTradingEnv is wrapped by DummyVecEnv and potentially VecNormalize
    # To get access to custom attributes like `self.data` or `self.portfolio_value` from MultiStockTradingEnv:
    # actual_env = env.envs[0] # If DummyVecEnv
    # if isinstance(env, VecNormalize):
    #   actual_env = env.venv.envs[0]
    # We need to be careful here. Let's try to get portfolio value from info dict first.

    total_steps = len(test_df) -1 # Minus one because env steps from 0 to len-2
    if hasattr(env, 'call') and 'get_attr' in env.getattr_methods: # Check if it's a VecEnv
      max_episode_steps = env.call('spec')[0].max_episode_steps if env.call('spec')[0] else float('inf')
    else: # Not a VecEnv or doesn't have typical spec
      max_episode_steps = float('inf') # Assume no limit if not found

    # Ensure total_steps doesn't exceed environment's max_episode_steps if defined
    # This is relevant if MAX_STEPS_PER_EPISODE_TEST is very large or not set effectively in create_env
    # total_steps = min(total_steps, max_episode_steps if max_episode_steps else total_steps)


    for i in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Portfolio value should be in info dict if MultiStockTradingEnv provides it
        # The info is for the vectorized env; need to extract for the single env
        current_portfolio_value = info[0].get('portfolio_value', INITIAL_AMOUNT)
        portfolio_values.append(current_portfolio_value)

        if i > 0 and portfolio_values[-2] != 0:
            daily_return = (current_portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)

        if done[0] or truncated[0] : # done and truncated are arrays for VecEnv
            print(f"Episode finished after {i+1} steps.")
            break
            # For testing on a fixed dataset, we usually run for the whole dataset length,
            # unless the environment itself terminates early (e.g. bankruptcy).
            # The `max_steps_per_episode` in `create_env` for testing should be >= len(test_df)
            # if we want to evaluate on the whole period.
            # For now, `MAX_STEPS_PER_EPISODE_TEST` might cut it short.

    return portfolio_values, daily_returns

def calculate_metrics(portfolio_values, daily_returns, risk_free_rate=0.0):
    print("\n--- Performance Metrics ---")
    if not portfolio_values or len(portfolio_values) < 2:
        print("Not enough data to calculate metrics.")
        return {}, pd.Series(dtype=float) # Return empty series for portfolio_df

    portfolio_df = pd.Series(portfolio_values)
    returns_df = pd.Series(daily_returns)

    # Cumulative Returns
    total_return = (portfolio_df.iloc[-1] - portfolio_df.iloc[0]) / portfolio_df.iloc[0] if portfolio_df.iloc[0] !=0 else 0

    # Max Drawdown
    peak = portfolio_df.expanding(min_periods=1).max()
    drawdown = (portfolio_df - peak) / peak.replace(0, pd.NA) # Avoid division by zero if peak is 0
    max_drawdown = drawdown.min()

    # Sharpe Ratio (annualized)
    # Assuming daily returns. Annualize by sqrt(252) (trading days)
    if not returns_df.empty and returns_df.std() != 0:
        sharpe_ratio = (returns_df.mean() - risk_free_rate/252) / returns_df.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    metrics = {
        "Total Return": f"{total_return:.2%}",
        "Max Drawdown": f"{max_drawdown:.2%}" if pd.notna(max_drawdown) else "N/A",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Final Portfolio Value": f"${portfolio_df.iloc[-1]:,.2f}",
        "Average Daily Return": f"{returns_df.mean():.4%}" if not returns_df.empty else "N/A",
        "Std Dev Daily Return": f"{returns_df.std():.4%}" if not returns_df.empty else "N/A",
    }

    print("\n--- Summary ---")
    for key, value in metrics.items():
        print(f"{key:<25}: {value}")

    return metrics, portfolio_df # Return portfolio_df for plotting

def plot_performance(portfolio_df):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df.values)
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)

        plot_filename = "portfolio_performance.png"
        plt.savefig(plot_filename)
        print(f"\nPlot saved to {plot_filename}")
        # plt.show() # This would block in a script; saving is better.
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error during plotting: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test a trained stock trading RL model.")
    parser.add_argument("--model", type=str, help="Path to the model .zip file. If not provided, lists available models.")
    parser.add_argument("--skip_plot", action="store_true", help="Skip generating performance plot.")

    args = parser.parse_args()

    selected_model_info = None
    if args.model:
        model_path = args.model
        stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        if not os.path.exists(stats_path):
            print(f"Warning: VecNormalize file not found at {stats_path}. Model might require it.")
            stats_path = None
        selected_model_info = {"model_path": model_path, "stats_path": stats_path}
    else:
        selected_model_info = list_models(MODEL_SAVE_DIR)

    if not selected_model_info:
        return

    # --- 1. Load Test Data ---
    # Use the same date ranges as eval_df in train.py
    # Tickers should ideally be part of the model's metadata or config.
    # For now, using DEFAULT_TICKER_LIST from train.py
    print(f"\n--- Preparing Test Data (using dates {DEFAULT_END_DATE} to {DEFAULT_TEST_END_DATE}) ---")

    # We need a way to determine num_stocks for the loaded model.
    # This is tricky if not saved with the model.
    # For now, we assume the test data uses DEFAULT_TICKER_LIST.
    # A robust solution would involve saving num_stocks or ticker_list with the model.

    # Path for the test data CSV
    # Ensure tickers in filename are sorted for consistency if DEFAULT_TICKER_LIST can change order elsewhere
    sorted_ticker_list_str = "_".join(sorted(DEFAULT_TICKER_LIST))
    test_data_filename = f"test_combined_data_{sorted_ticker_list_str}_{DEFAULT_END_DATE}_{DEFAULT_TEST_END_DATE}.csv"
    test_data_path = os.path.join(DATA_DIR, test_data_filename)

    if os.path.exists(test_data_path):
        print(f"Loading existing test data from: {test_data_path}")
        test_df = pd.read_csv(test_data_path)
        test_df["Date"] = pd.to_datetime(test_df["Date"])
        test_df.set_index("Date", inplace=True)
        num_stocks_for_env = len(DEFAULT_TICKER_LIST) # Assuming this matches the data
    else:
        print(f"Test data not found at {test_data_path}. Generating it now...")
        # Call prepare_data from train.py - it creates both train and eval(test) data.
        # We only need the eval_df (test_df here).
        # prepare_data returns: train_df, eval_df, num_tickers
        try:
            # This part is a bit tricky because prepare_data generates both train and test files.
            # We're interested in the test file (eval_df from prepare_data's perspective).
            # The train_start_date and train_end_date for prepare_data can be set to minimal values
            # if we only care about the test data generation part.
            # Using DEFAULT_END_DATE for both train_start and train_end means the "training" part
            # of prepare_data will process a very small (or possibly empty if dates are exclusive) dataset.
            print(f"Calling prepare_data with train_start={DEFAULT_END_DATE}, train_end={DEFAULT_END_DATE}, test_end={DEFAULT_TEST_END_DATE}")
            _, test_df, num_stocks_for_env = prepare_data(
                DEFAULT_TICKER_LIST,
                DEFAULT_END_DATE, # train_start_date for prepare_data
                DEFAULT_END_DATE, # train_end_date for prepare_data
                DEFAULT_TEST_END_DATE # test_end_date for prepare_data
            )
            print(f"Test data generated successfully by prepare_data. Shape: {test_df.shape}")
        except Exception as e:
            print(f"Error generating test data using train.prepare_data: {e}")
            print("Please ensure that the data preprocessing pipeline can run.")
            print("You might need to run train.py once to generate the test data if this fails.")
            return

    if test_df.empty:
        print("Test data is empty. Cannot proceed.")
        return

    # --- 2. Load Model and Environment ---
    try:
        # The num_stocks_for_env is critical here.
        # It should match the number of stocks the model was trained on.
        # This info should ideally be saved with the model or inferred from its architecture.
        # For now, we derive it from DEFAULT_TICKER_LIST used for test data generation.
        model, test_env = load_model_and_env(selected_model_info, test_df, num_stocks_for_env)
    except Exception as e:
        print(f"Error loading model or environment: {e}")
        # This can happen if AlgoClass is wrong, or env mismatches model.
        raise # Reraise to see traceback for debugging
        # return # Unreachable due to raise

    # --- 3. Run Evaluation ---
    portfolio_values, daily_returns = run_evaluation(model, test_env, test_df)

    # --- 4. Calculate and Display Metrics ---
    metrics, portfolio_df_series = calculate_metrics(portfolio_values, daily_returns)

    # --- 5. Optional Plotting ---
    if not args.skip_plot:
        if not portfolio_df_series.empty:
            plot_performance(portfolio_df_series)
        else:
            print("Portfolio data is empty, skipping plot.")

    print("\n--- Testing Finished ---")

if __name__ == "__main__":
    # Ensure dummy files for DataPreprocessingPipeline components exist for prepare_data call
    # This is a simplified version of what train.py does.
    # If train.py has run successfully, these may not be strictly necessary
    # if the actual components are functional.
    # However, to make test.py runnable independently for basic checks if data is missing:
    required_preprocessing_files = [
        "src/data_preprocessing/CollectData.py",
        "src/data_preprocessing/PreProcess.py",
        "src/data_preprocessing/FeatureEngineer.py",
        "src/data_preprocessing/Post_Process_Features.py",
        "src/data_preprocessing/Download_Macro.py",
        "src/data_preprocessing/PreProcessMacro.py",
        "src/data_preprocessing/CombineDf.py"
    ]
    # This logic was in train.py; it's good practice for test.py to not mandate it
    # and assume that if prepare_data is called, the actual (or sufficiently dummied)
    # pipeline components are available. The previous subtask (test run of train.py)
    # already created more robust dummies.
    # For this subtask, we are just creating test.py structure.
    # The dummy file creation in the original prompt for test.py was very minimal.
    # We will rely on the dummies created in the previous "test run of train.py" task.
    main()
