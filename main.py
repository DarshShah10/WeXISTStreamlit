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
        from train import run_training_programmatically

        tickers_list = [ticker.strip().upper() for ticker in args.tickers.split(',') if ticker.strip()]
        if not tickers_list:
            print("Error: No tickers provided for training.")
            return

        print(f"Algorithm: {args.algo}")
        print(f"Tickers: {tickers_list}")
        print(f"Training Start Date: {args.start_date}")
        print(f"Training End Date: {args.end_date}")
        print(f"Test End Date: {args.test_end_date}")
        print(f"Total Timesteps: {args.timesteps}")
        if args.tag:
            print(f"Model Tag: {args.tag}")

        saved_model_path = run_training_programmatically(
            selected_algo_name_param=args.algo,
            ticker_list_param=tickers_list,
            start_date_param=args.start_date,
            end_date_param=args.end_date,
            test_end_date_param=args.test_end_date,
            total_timesteps_param=args.timesteps,
            model_tag_param=args.tag if args.tag else ""
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
    """Handles the model evaluation sub-command."""
    print("Model evaluation subcommand initiated.")
    try:
        # Construct the command to call test.py
        # Ensure test.py is executable and uses python from the environment
        # sys.executable provides the path to the current Python interpreter
        command = [
            sys.executable,
            "test.py",
            "--model_path", args.model_path,
            "--data_path", args.data_path,
            "--tickers", args.tickers
        ]

        print(f"Running evaluation command: {' '.join(command)}")

        # We use subprocess.run and capture output.
        # Check=True will raise CalledProcessError if test.py exits with non-zero status.
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("\n--- Evaluation Output ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- Evaluation Errors (if any) ---")
            print(result.stderr)
        print("Model evaluation finished.")

    except FileNotFoundError:
        print("Error: test.py not found in the project root or python interpreter not found.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running test.py: {e}")
        print("\n--- test.py STDOUT ---")
        print(e.stdout)
        print("\n--- test.py STDERR ---")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during model evaluation: {e}")

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
    parser_train.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock tickers for training.")
    parser_train.add_argument("--start_date", type=str, required=True, help="Training start date (YYYY-MM-DD).")
    parser_train.add_argument("--end_date", type=str, required=True, help="Training end date (YYYY-MM-DD).")
    parser_train.add_argument("--test_end_date", type=str, required=True, help="Test data end date (YYYY-MM-DD) for evaluation split during training.")
    parser_train.add_argument("--timesteps", type=int, required=True, help="Total training timesteps.")
    parser_train.add_argument("--tag", type=str, help="Optional model name tag (e.g., 'my_custom_run').")
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
