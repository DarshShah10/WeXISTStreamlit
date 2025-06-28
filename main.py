import argparse
import os
import subprocess
import sys
from datetime import datetime

# Ensure the src directory is in the Python path
# This allows direct imports from src.data_preprocessing, train, test etc.
# when main.py is run from the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
# If your modules are in a 'src' subdirectory, you might need:
# sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
# However, current imports like `from train import ...` suggest train.py is at root
# or Python path is already configured for it.
# Let's assume current structure allows imports like `from train import ...`
# and `from src.data_preprocessing...` if those modules are structured accordingly.

# It's good practice to only import when needed, especially for subcommands.

def handle_process_data(args):
    """Handles the data processing sub-command."""
    print("Data processing subcommand initiated.")
    try:
        from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline

        tickers_list = [ticker.strip().upper() for ticker in args.tickers.split(',') if ticker.strip()]
        if not tickers_list:
            print("Error: No tickers provided.")
            return

        pipeline = StockPreProcessPipeline()
        print(f"Running data pipeline for tickers: {tickers_list}")
        print(f"Start date: {args.start_date}, End date: {args.end_date}")
        print(f"Output path: {args.output_path}")

        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        processed_file_path = pipeline.run_data_pipeline(
            ticker_list=tickers_list,
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            output_csv_path=args.output_path
        )
        print(f"Data processing successful. Output saved to: {processed_file_path}")
    except ImportError:
        print("Error: Could not import StockPreProcessPipeline. Make sure it's correctly placed and PYTHONPATH is set.")
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        print("Ensure all dependencies for the pipeline are installed and paths are correct.")

def handle_train_model(args):
    """Handles the model training sub-command."""
    print("Model training subcommand initiated.")
    try:
        from train import run_training_programmatically # Assuming CONFIG is loaded within train.py
        from src.utils import load_config # To load config here for use_fundamental_tickers logic if needed by train.py

        config = load_config() # Load config for train.py

        tickers_list = []
        if args.tickers:
            tickers_list = [ticker.strip().upper() for ticker in args.tickers.split(',') if ticker.strip()]

        if args.use_fundamental_tickers:
            print("Flag --use_fundamental_tickers is set. Ticker list will be determined by the fundamental analysis module.")
            # The tickers_list can be empty here; run_training_programmatically will handle it.
            if tickers_list:
                print(f"Warning: --tickers argument ({tickers_list}) provided but will be ignored due to --use_fundamental_tickers.")
                tickers_list = [] # Ensure it's empty so fundamental tickers are definitely used
        elif not tickers_list:
            print("Error: No tickers provided and --use_fundamental_tickers is not set. Please provide --tickers or use --use_fundamental_tickers.")
            return

        print(f"Algorithm: {args.algo}")
        if tickers_list: # Only print if tickers were manually provided and not overridden
            print(f"Tickers (manual): {tickers_list}")
        print(f"Training Start Date: {args.start_date}")
        print(f"Training End Date: {args.end_date}")
        print(f"Test End Date: {args.test_end_date}")
        print(f"Total Timesteps: {args.timesteps}")
        if args.tag:
            print(f"Model Tag: {args.tag}")

        saved_model_path = run_training_programmatically(
            config=config, # Pass the loaded config object
            selected_algo_name=args.algo, # Parameter name in train.py might be selected_algo_name
            ticker_list=tickers_list, # Parameter name in train.py might be ticker_list
            start_date=args.start_date, # Parameter name in train.py might be start_date
            end_date=args.end_date, # Parameter name in train.py might be end_date
            test_end_date=args.test_end_date, # Parameter name in train.py might be test_end_date
            total_timesteps=args.timesteps, # Parameter name in train.py might be total_timesteps
            model_tag=args.tag if args.tag else "", # Parameter name in train.py might be model_tag
            use_fundamental_tickers=args.use_fundamental_tickers # Pass the new flag
        )
        if saved_model_path:
            print(f"Model training successful. Model saved to: {saved_model_path}")
        else:
            print("Model training may have failed or the algorithm was not found. Check logs.")
    except ImportError:
        print("Error: Could not import run_training_programmatically from train.py.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")

def handle_evaluate_model(args):
    """Handles the model evaluation sub-command using the BaseBacktester framework."""
    print("Model evaluation subcommand initiated.")
    try:
        from src.utils import load_config
        from src.envs.backtest import TD3Backtester
        from src.envs.backtestppo import RecurrentPPOBacktester
        # Add other backtesters as needed, e.g. PPOBacktester, A2CBacktester

        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            return
        if not os.path.exists(args.data_path):
            print(f"Error: Data file not found at {args.data_path}")
            return

        config = load_config() # Load shared configuration
        tickers_list = [ticker.strip().upper() for ticker in args.tickers.split(',') if ticker.strip()]
        if not tickers_list:
            print("Error: No tickers provided for evaluation.")
            return

        # Infer model type from model_path to select the correct backtester
        # This is a heuristic. A more robust way might be to store model type with the model
        # or have the user specify it.
        model_filename_lower = os.path.basename(args.model_path).lower()
        BacktesterClass = None
        if "td3" in model_filename_lower:
            BacktesterClass = TD3Backtester
        elif "ppo" in model_filename_lower: # Catches both PPO and RecurrentPPO if names are consistent
            # Differentiate between PPO and RecurrentPPO if necessary based on model or config structure
            # For now, let's assume RecurrentPPO if "recurrent" or "lstm" is in the name, else standard PPO.
            # This part might need refinement based on how models are named or if a generic PPOBacktester is created.
            if "recurrent" in model_filename_lower or "lstm" in model_filename_lower:
                 BacktesterClass = RecurrentPPOBacktester
            else:
                 # If you have a separate PPOBacktester for non-recurrent PPO:
                 # from src.envs.backtest_ppo_standard import PPOBacktester # Example
                 # BacktesterClass = PPOBacktester
                 # For now, falling back to RecurrentPPOBacktester for any "ppo"
                 print("Warning: Inferred PPO model. Using RecurrentPPOBacktester. Create a specific PPOBacktester if this is not a recurrent model.")
                 BacktesterClass = RecurrentPPOBacktester # Or a future PPOBacktester
        # Add elif for A2C, DDPG etc.
        # elif "a2c" in model_filename_lower:
        #     from src.envs.backtest_a2c import A2CBacktester # Assuming it exists
        #     BacktesterClass = A2CBacktester
        else:
            print(f"Error: Could not determine backtester type from model name: {args.model_path}")
            print("Please ensure model name contains 'td3', 'ppo', 'recurrentppo', etc.")
            return

        print(f"Using Backtester: {BacktesterClass.__name__}")

        # Path for VecNormalize stats (if saved alongside the model)
        env_stats_path = args.model_path.replace(config.get('model_zip_suffix', '.zip'), config.get('vecnormalize_suffix', '_vecnormalize.pkl'))
        if not os.path.exists(env_stats_path):
            print(f"Warning: VecNormalize stats file not found at {env_stats_path}. Proceeding without it.")
            env_stats_path = None

        backtester = BacktesterClass(
            model_path=args.model_path,
            env_stats_path=env_stats_path,
            config=config,
            ticker_list=tickers_list,
            output_dir_suffix="cli_eval" # Add a suffix to distinguish CLI evaluations
        )

        print(f"Starting backtest for model: {args.model_path} with data: {args.data_path}")
        metrics, results_df = backtester.run_full_backtest(backtest_data_source=args.data_path)

        print("\n--- Evaluation Metrics ---")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # The backtester already saves plots and detailed results to its output directory.
        # We can print the path to that directory.
        print(f"\nDetailed results, plots, and metrics saved in: {backtester.output_dir}")
        print("Model evaluation finished.")

    except ImportError as e:
        print(f"Error importing necessary modules for evaluation: {e}")
    except FileNotFoundError as e: # Should be caught by direct checks now
        print(f"Error: A required file was not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during model evaluation: {e}")
        import traceback
        traceback.print_exc()


def handle_launch_app(args):
    """Handles the Streamlit app launching sub-command."""
    print("Launching Streamlit application...")
    try:
        command = ["streamlit", "run", "streamlit_app.py"]
        print(f"Executing: {' '.join(command)}")
        # For Streamlit, it's better to let it run in the foreground
        # and not capture output unless for specific debugging.
        # Using Popen allows it to run more like a typical Streamlit launch.
        process = subprocess.Popen(command)
        process.wait() # Wait for the app to be closed by the user
    except FileNotFoundError:
        print("Error: streamlit command not found. Make sure Streamlit is installed and in PATH.")
        print("You can install it with: pip install streamlit")
    except Exception as e:
        print(f"An error occurred while launching the Streamlit app: {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for the RL Stock Trading Bot Project.")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands", required=True)

    # Sub-parser for Data Processing
    parser_process_data = subparsers.add_parser("process_data", help="Run the data preprocessing pipeline.")
    parser_process_data.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock tickers (e.g., 'AAPL,MSFT,GOOG').")
    parser_process_data.add_argument("--start_date", type=str, required=True, help="Start date for data collection (YYYY-MM-DD).")
    parser_process_data.add_argument("--end_date", type=str, required=True, help="End date for data collection (YYYY-MM-DD).")
    parser_process_data.add_argument("--output_path", type=str, required=True, help="Full file path to save the processed CSV data (e.g., 'data/processed/my_data.csv').")
    parser_process_data.set_defaults(func=handle_process_data)

    # Sub-parser for Model Training
    parser_train = subparsers.add_parser("train", help="Train a new reinforcement learning model.")
    parser_train.add_argument("--algo", type=str, required=True, choices=["PPO", "A2C", "TD3", "DDPG", "RecurrentPPO"], help="Algorithm to use for training.")
    parser_train.add_argument("--tickers", type=str, help="Comma-separated list of stock tickers for training. Optional if --use_fundamental_tickers is set.")
    parser_train.add_argument("--start_date", type=str, required=True, help="Training start date (YYYY-MM-DD).")
    parser_train.add_argument("--end_date", type=str, required=True, help="Training end date (YYYY-MM-DD).")
    parser_train.add_argument("--test_end_date", type=str, required=True, help="Test data end date (YYYY-MM-DD) for evaluation split during training.")
    parser_train.add_argument("--timesteps", type=int, required=True, help="Total training timesteps.")
    parser_train.add_argument("--tag", type=str, help="Optional model name tag (e.g., 'my_custom_run').")
    parser_train.add_argument("--use_fundamental_tickers", action="store_true", help="Use tickers from fundamental analysis instead of providing a list.")
    parser_train.set_defaults(func=handle_train_model)

    # Sub-parser for Model Evaluation
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    parser_evaluate.add_argument("--model_path", type=str, required=True, help="Path to the trained model .zip file.")
    parser_evaluate.add_argument("--data_path", type=str, required=True, help="Path to the evaluation data CSV file.")
    parser_evaluate.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock tickers relevant to the data file.")
    parser_evaluate.set_defaults(func=handle_evaluate_model)

    # Sub-parser for Launching Streamlit App
    parser_app = subparsers.add_parser("app", help="Launch the Streamlit application.")
    parser_app.set_defaults(func=handle_launch_app)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    # This simple check is a placeholder for more complex path management if needed.
    # For example, if main.py was in a 'scripts' folder, PROJECT_ROOT logic would be more critical.
    # print(f"Project root (based on main.py location): {PROJECT_ROOT}")
    # print(f"Python sys.path[0]: {sys.path[0]}")
    main()
