import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta # Ensure datetime is imported from datetime
import plotly.express as px
import plotly.graph_objects as go
import logging # Optional: for logging config loading issues

# Stable Baselines3 components
from stable_baselines3 import PPO, A2C, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Project-specific imports
from src.utils import load_config # Import the config loader
from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from src.envs.trading_env import MultiStockTradingEnv

# Import functions from train.py (now config-driven)
from train import create_env, run_training_programmatically
# Removed imports from test.py as BaseBacktester handles metrics/plots
# from test import calculate_metrics, plot_performance, run_evaluation

# Import new backtester classes
from src.envs.backtest import TD3Backtester
from src.envs.backtestppo import RecurrentPPOBacktester
# Add other specific backtesters if needed, e.g. from src.envs.backtesta2c import A2CBacktester

# --- Global Configuration & Session State Initialisation ---
if 'config' not in st.session_state:
    try:
        st.session_state.config = load_config()
        # PROJECT_ROOT is already part of the config loaded by load_config
    except Exception as e:
        logging.error(f"Critical: Failed to load configuration at Streamlit app start: {e}")
        st.error(f"CRITICAL ERROR: Could not load configuration. App cannot start. Details: {e}")
        # Stop further execution if config fails, as it's critical
        st.stop()

CONFIG = st.session_state.config # Use config from session state
PROJECT_ROOT = CONFIG.get('project_root', os.path.dirname(os.path.abspath(__file__)))


# Use paths from CONFIG.
# If not, construct them using PROJECT_ROOT. For now, assume they are absolute or correctly relative.
MODEL_SAVE_DIR = CONFIG.get('model_dir', os.path.join(PROJECT_ROOT, 'models'))
DATA_DIR = CONFIG.get('data_dir', os.path.join(PROJECT_ROOT, 'data'))
APP_TEMP_DATA_DIR = os.path.join(DATA_DIR, "app_temp") # Specific for app's temporary data files


# Helper to convert date string from config to datetime object
def _parse_date_from_config(date_str, default_offset_days=0):
    if date_str:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            logging.warning(f"Invalid date format in config: {date_str}. Using default.")
    return datetime.now() - timedelta(days=default_offset_days)


@st.cache_data
def list_streamlit_models(_model_dir_from_config): # Argument name changed to avoid conflict with global
    models = []
    # Search for models directly under model_dir (e.g., PPO_tag_timestamp.zip)
    # and also under model_dir/algo_tag_timestamp/algo_tag.zip (newer structure from train.py)

    # Pattern 1: model_dir/*.zip
    zip_files_pattern1 = glob.glob(os.path.join(_model_dir_from_config, f"*{CONFIG.get('model_zip_suffix', '.zip')}"))
    for model_path in zip_files_pattern1:
        stats_path = model_path.replace(CONFIG.get('model_zip_suffix', '.zip'), CONFIG.get('vecnormalize_suffix', '_vecnormalize.pkl'))
        models.append({
            "name": os.path.basename(model_path),
            "model_path": model_path,
            "stats_path": stats_path if os.path.exists(stats_path) else None
        })

    # Pattern 2: model_dir/RUN_SPECIFIC_MODEL_DIR/MODEL_NAME.zip (e.g. models/ppo_mytag_20231026_103000/ppo_mytag.zip)
    # Also check for models saved directly in subdirectories of model_dir (run_specific_model_artifacts_dir)
    sub_dirs = glob.glob(os.path.join(_model_dir_from_config, "*", "")) # Get all subdirectories
    for sub_dir in sub_dirs:
        # Check for model zip files within this subdirectory
        # The model zip file name is often the same as the subdirectory name (without timestamp) or a fixed name.
        # For example, if subdir is "ppo_cli_run_20231115_100000", model might be "ppo_cli_run.zip"

        # Try to find a zip file in the subdir that matches parts of the subdir name or a generic name
        potential_model_files = glob.glob(os.path.join(sub_dir, f"*{CONFIG.get('model_zip_suffix', '.zip')}"))
        for model_path in potential_model_files:
             stats_path = model_path.replace(CONFIG.get('model_zip_suffix', '.zip'), CONFIG.get('vecnormalize_suffix', '_vecnormalize.pkl'))
             # Display name relative to the main model directory for clarity
             display_name = os.path.relpath(model_path, _model_dir_from_config)
             # Avoid adding duplicates if already found by pattern 1 (less likely with new structure)
             if not any(m['model_path'] == model_path for m in models):
                models.append({
                    "name": display_name, # e.g. "ppo_mytag_timestamp/ppo_mytag.zip"
                    "model_path": model_path,
                    "stats_path": stats_path if os.path.exists(stats_path) else None
                })

    # Pattern 3: Best models (model_dir/best_model/RUN_SPECIFIC_DIR/best_model.zip)
    best_model_base_dir = CONFIG.get('best_model_dir', os.path.join(_model_dir_from_config, "best_model"))
    if os.path.exists(best_model_base_dir):
        best_model_run_specific_dirs = glob.glob(os.path.join(best_model_base_dir, "*", "")) # e.g. models/best_model/ppo_mytag_timestamp/
        for best_model_run_dir in best_model_run_specific_dirs:
            best_model_path = os.path.join(best_model_run_dir, f"best_model{CONFIG.get('model_zip_suffix', '.zip')}")
            # Stats path for best models is often saved as vecnormalize.pkl or best_model_vecnormalize.pkl
            best_model_stats_path = os.path.join(best_model_run_dir, f"vecnormalize{CONFIG.get('vecnormalize_suffix', '.pkl')}") # From EvalCallback
            if not os.path.exists(best_model_stats_path): # try alternative name
                 best_model_stats_path = os.path.join(best_model_run_dir, f"best_model{CONFIG.get('vecnormalize_suffix', '.pkl')}")

            if os.path.exists(best_model_path):
                display_name = os.path.relpath(best_model_path, _model_dir_from_config)
                if not any(m['model_path'] == best_model_path for m in models):
                    models.append({
                        "name": display_name, # e.g., "best_model/ppo_mytag_timestamp/best_model.zip"
                        "model_path": best_model_path,
                        "stats_path": best_model_stats_path if os.path.exists(best_model_stats_path) else None
                    })
    return sorted(list(set(m['name'] for m in models))) # Return unique names, then reconstruct full dict if needed, or return full models list
    # Returning full models list is better
    unique_models = []
    seen_paths = set()
    for model_info in models:
        if model_info['model_path'] not in seen_paths:
            unique_models.append(model_info)
            seen_paths.add(model_info['model_path'])
    return sorted(unique_models, key=lambda x: x['name'])


def load_streamlit_model_and_env(selected_model_info, data_df, num_stocks_env, app_config):
    model_path = selected_model_info["model_path"]
    stats_path = selected_model_info["stats_path"]

    st.write(f"Loading Model: {model_path}")

    model_name_lower = os.path.basename(model_path).lower()
    parent_dir_name = os.path.basename(os.path.dirname(model_path)).lower()

    AlgoClass = None
    # Try to infer from filename first, then from parent directory (for best_model cases)
    algo_hints = [model_name_lower, parent_dir_name]
    for hint in algo_hints:
        if "ppo" in hint: AlgoClass = PPO; break
        elif "a2c" in hint: AlgoClass = A2C; break
        elif "td3" in hint: AlgoClass = TD3; break
        elif "ddpg" in hint: AlgoClass = DDPG; break

    if AlgoClass is None:
        st.warning(f"Could not infer algorithm type from model name '{model_name_lower}' or parent dir '{parent_dir_name}'. Attempting PPO as default.")
        AlgoClass = PPO

    # create_env is imported from train.py, which now uses its own global CONFIG.
    # We pass app_config (which is CONFIG here) to it, as train.py's create_env expects it.
    # No need to manipulate train module's globals here.
    raw_env = create_env(
        df=data_df,
        num_stocks_env=num_stocks_env,
        config=app_config, # Pass the global config
        training=False # Critical for inference/backtesting
    )
    vec_env = DummyVecEnv([lambda: raw_env])

    if stats_path and os.path.exists(stats_path):
        st.write(f"Loading VecNormalize stats from: {stats_path}")
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False # Ensure it's set to False for inference
        vec_env.norm_reward = False # Typically, reward normalization is not applied during inference
    else:
        st.warning("No VecNormalize stats found or path is incorrect. Using unnormalized environment. This might affect performance if the model was trained with normalization.")

    model = AlgoClass.load(model_path, env=vec_env)
    st.success("Model loaded successfully.")
    return model, vec_env

st.set_page_config(page_title=CONFIG.get("streamlit_app_title", "RL Trading Dashboard"), layout="wide", page_icon="📈")
st.title(f"🤖 {CONFIG.get('streamlit_app_title', 'Reinforcement Learning Trading Dashboard')}")


# Initialize session state variables if they don't exist
# This helps maintain state across reruns and tabs
if "selected_model_info" not in st.session_state: st.session_state.selected_model_info = None
# Remove old model and vec_env from session state as BaseBacktester handles this internally for its run
# if "model" not in st.session_state: st.session_state.model = None
# if "vec_env" not in st.session_state: st.session_state.vec_env = None
if "processed_data_df_for_backtest" not in st.session_state: st.session_state.processed_data_df_for_backtest = None # More specific name
if "backtest_metrics_dict" not in st.session_state: st.session_state.backtest_metrics_dict = None # For metrics dict
if "backtest_results_df" not in st.session_state: st.session_state.backtest_results_df = None # For results df
if "processed_train_data_path" not in st.session_state: st.session_state.processed_train_data_path = None

st.sidebar.header("⚙️ Configuration")

# Use MODEL_SAVE_DIR which is now derived from CONFIG
available_models_list = list_streamlit_models(MODEL_SAVE_DIR)

if not available_models_list:
    st.sidebar.warning(f"No trained models found in '{MODEL_SAVE_DIR}'. Check config variable 'model_dir'.")
else:
    model_names = [m["name"] for m in available_models_list]
    selected_model_name = st.sidebar.selectbox(
        "Select Trained Model",
        options=model_names,
        index=0 if model_names else -1,
        key="sb_select_model_name"
    )
    if selected_model_name:
        st.session_state.selected_model_info = next((m for m in available_models_list if m["name"] == selected_model_name), None)


st.sidebar.subheader("Data for Backtest/Inference")
# Use default tickers from CONFIG
default_tickers_list = CONFIG.get('default_ticker_list', ["AAPL", "MSFT", "GOOG"])
user_tickers_str = st.sidebar.text_input(
    "Enter Stock Tickers (comma-separated)",
    value=", ".join(default_tickers_list[:3]) # Show first 3 by default
)
selected_tickers = [ticker.strip().upper() for ticker in user_tickers_str.split(",") if ticker.strip()]

# Use default dates from CONFIG, convert to datetime
# Fallback to relative dates if config values are missing/invalid
config_start_date_str = CONFIG.get('default_start_date')
config_end_date_str = CONFIG.get('default_end_date') # This is usually train end, for app might mean recent data

# For sidebar backtest range, let's use a longer history by default
sidebar_default_start = _parse_date_from_config(config_start_date_str, 3*365) # Default 3 years ago
sidebar_default_end = _parse_date_from_config(config_end_date_str, 1) # Default yesterday

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=sidebar_default_start, key="sb_start_date")
with col2:
    end_date = st.date_input("End Date", value=sidebar_default_end, key="sb_end_date")

# Main application tabs
tab_data_processing, tab_model_training, tab_inference_backtesting, tab_model_info = st.tabs([
    "🛠️ Data Processing",
    "🧠 Model Training",
    "📊 Inference & Backtesting",
    "ℹ️ Model & Environment Info"
])

with tab_data_processing:
    st.header("Data Processing Pipeline")
    st.markdown("Process historical market data for training or analysis.")

    dp_default_tickers = CONFIG.get('default_ticker_list', ["RELIANCE.NS", "TCS.NS"])
    dp_user_tickers_str = st.text_input(
        "Stock Tickers for Processing (comma-separated)",
        value=", ".join(dp_default_tickers),
        key="dp_tickers"
    )
    dp_selected_tickers = [ticker.strip().upper() for ticker in dp_user_tickers_str.split(",") if ticker.strip()]

    # Default dates for data processing (typically longer for training sets)
    dp_default_start = _parse_date_from_config(CONFIG.get('default_start_date'), 5*365) # 5 years for training data
    dp_default_end = _parse_date_from_config(CONFIG.get('default_end_date'), 1) # Default to yesterday (end of available data)

    dp_col1, dp_col2 = st.columns(2)
    with dp_col1:
        dp_start_date = st.date_input("Start Date for Processing", value=dp_default_start, key="dp_start_date")
    with dp_col2:
        dp_end_date = st.date_input("End Date for Processing", value=dp_default_end, key="dp_end_date")

    if st.button("⚙️ Process Market Data", key="btn_process_market_data"):
        if not dp_selected_tickers:
            st.warning("Please enter at least one stock ticker for processing.")
        elif dp_start_date >= dp_end_date:
            st.error("Error: Start date must be before end date for processing.")
        else:
            with st.spinner(f"Processing data for: {', '.join(dp_selected_tickers)}..."):
                try:
                    os.makedirs(APP_TEMP_DATA_DIR, exist_ok=True)
                    sorted_dp_tickers_str = "_".join(sorted(dp_selected_tickers))
                    output_filename = f"processed_app_data_{sorted_dp_tickers_str}_{dp_start_date.strftime('%Y%m%d')}_{dp_end_date.strftime('%Y%m%d')}.csv"
                    output_csv_path = os.path.join(APP_TEMP_DATA_DIR, output_filename)

                    # Instantiate pipeline with global CONFIG
                    pipeline = StockPreProcessPipeline(config=CONFIG)
                    processed_path = pipeline.run_data_pipeline(
                        dp_selected_tickers,
                        dp_start_date.strftime('%Y-%m-%d'),
                        dp_end_date.strftime('%Y-%m-%d'),
                        output_csv_path
                    )
                    st.session_state.processed_train_data_path = processed_path
                    st.success(f"Market data processed! Saved to: {processed_path}")
                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    st.exception(e)

    if st.session_state.processed_train_data_path:
        st.markdown(f"**Last processed data path (for potential training):** `{st.session_state.processed_train_data_path}`")


with tab_model_training:
    st.header("Train a New RL Model")
    st.markdown("Configure parameters and start training. Model will be saved based on `config/shared_config.yaml` paths.")

    train_algo_options = list(CONFIG.get('ppo', {}).keys()) # Example: Get keys from PPO, or define explicitly
    # A more robust way: list known algos, check if config exists
    defined_algos = ["PPO", "A2C", "TD3", "DDPG", "RecurrentPPO"]
    train_algo_options = [algo for algo in defined_algos if algo.lower() in CONFIG]
    if not train_algo_options: train_algo_options = defined_algos # Fallback if config sections are missing

    train_selected_algo = st.selectbox("Select Algorithm", options=train_algo_options, index=0, key="train_algo_select")

    train_default_tickers = CONFIG.get('default_ticker_list', ["RELIANCE.NS", "TCS.NS"])
    train_tickers_str = st.text_input(
        "Stock Tickers for Training",
        value=", ".join(train_default_tickers),
        key="train_tickers"
    )
    train_selected_tickers = [ticker.strip().upper() for ticker in train_tickers_str.split(",") if ticker.strip()]

    # Default dates for training from config
    train_def_start_date = _parse_date_from_config(CONFIG.get('default_start_date'), 4*365)
    train_def_end_date = _parse_date_from_config(CONFIG.get('default_end_date'), 1*365) # End of training
    train_def_test_end_date = _parse_date_from_config(CONFIG.get('default_test_end_date'), 1) # End of test/eval

    train_col1, train_col2, train_col3 = st.columns(3)
    with train_col1:
        train_start_date = st.date_input("Training Start Date", value=train_def_start_date, key="train_start_date")
    with train_col2:
        train_end_date = st.date_input("Training End Date (for train data)", value=train_def_end_date, key="train_end_date")
    with train_col3:
        train_test_end_date = st.date_input("Test Data End Date (for eval)", value=train_def_test_end_date, key="train_test_end_date")

    train_total_timesteps = st.number_input(
        "Total Training Timesteps",
        min_value=1000,
        value=CONFIG.get('default_total_timesteps', 50000),
        step=1000,
        key="train_timesteps"
    )
    train_model_tag = st.text_input("Model Name Tag (Optional)", key="train_model_tag", placeholder="e.g., my_ppo_run")

    if st.button("🚀 Start Training", key="btn_start_training"):
        # Validations... (similar to before)
        if not train_selected_tickers or not train_selected_algo or \
           train_start_date >= train_end_date or train_end_date >= train_test_end_date:
            st.warning("Please check inputs: tickers, algorithm, and date ranges.")
        else:
            st.info(f"Starting training for {train_selected_algo}...")
            with st.spinner(f"Training {train_selected_algo} model... Check console for logs."):
                try:
                    saved_model_path = run_training_programmatically(
                        config=CONFIG, # Pass the global CONFIG
                        selected_algo_name=train_selected_algo, # Note: param name changed in train.py
                        ticker_list=train_selected_tickers,      # Note: param name changed
                        start_date=train_start_date.strftime('%Y-%m-%d'), # Note: param name changed
                        end_date=train_end_date.strftime('%Y-%m-%d'),     # Note: param name changed
                        test_end_date=train_test_end_date.strftime('%Y-%m-%d'), # Note: param name changed
                        total_timesteps=int(train_total_timesteps), # Note: param name changed
                        model_tag=train_model_tag # Note: param name changed
                    )
                    if saved_model_path:
                        st.success(f"Training finished! Model saved to: {saved_model_path}")
                        st.cache_data.clear() # Clear model list cache
                        st.info("Model list updated. Refresh or reselect model in sidebar.")
                    else:
                        st.error("Training failed. Check console logs.")
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    st.exception(e)


with tab_inference_backtesting:
    st.header("Run Inference / Backtest")

    if not st.session_state.selected_model_info:
        st.warning("Please select a model from the sidebar.")
    elif not selected_tickers: # selected_tickers is from sidebar
        st.warning("Please enter at least one stock ticker in the sidebar.")
    elif start_date >= end_date: # start_date and end_date from sidebar
        st.error("Error: Start date must be before end date.")
    else:
        st.write(f"**Selected Model**: {st.session_state.selected_model_info['name']}")
        st.write(f"**Tickers**: {', '.join(selected_tickers)}")
        st.write(f"**Date Range**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        if st.button("🚀 Process Data & Run Backtest", key="btn_run_backtest"):
            st.session_state.processed_data_df_for_backtest = None
            st.session_state.backtest_metrics_dict = None
            st.session_state.backtest_results_df = None
            backtester_output_dir_for_run = None # To store and display path to backtester's unique output dir

            processed_data_csv_path = None
            with st.spinner("Processing data for backtesting..."):
                try:
                    os.makedirs(APP_TEMP_DATA_DIR, exist_ok=True)
                    sorted_selected_tickers_str = "_".join(sorted(selected_tickers))
                    output_filename = f"backtest_app_data_{sorted_selected_tickers_str}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    processed_data_csv_path = os.path.join(APP_TEMP_DATA_DIR, output_filename)

                    pipeline = StockPreProcessPipeline(config=st.session_state.config)
                    processed_data_csv_path = pipeline.run_data_pipeline(
                        selected_tickers,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        processed_data_csv_path
                    )
                    if processed_data_csv_path and os.path.exists(processed_data_csv_path):
                        st.success(f"Data processed successfully! Saved to: {processed_data_csv_path}")
                    else:
                        st.error("Data processing failed to produce an output file.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    st.exception(e)
                    st.stop()

            if processed_data_csv_path and os.path.exists(processed_data_csv_path):
                with st.spinner("Running backtest with selected model..."):
                    try:
                        model_info = st.session_state.selected_model_info
                        model_file_path = model_info['model_path']
                        model_name_lower = model_info['name'].lower()
                        env_stats = model_info['stats_path']

                        backtester = None
                        model_type_key = None
                        if "td3" in model_name_lower: model_type_key = "TD3"
                        elif "recurrentppo" in model_name_lower or "recurrent_ppo" in model_name_lower : model_type_key = "RecurrentPPO"
                        elif "ppo" in model_name_lower: model_type_key = "PPO" # Generic PPO
                        # Add A2C, DDPG later if needed

                        if model_type_key == "TD3":
                            backtester = TD3Backtester(
                                model_path=model_file_path, env_stats_path=env_stats,
                                config=st.session_state.config, ticker_list=selected_tickers
                            )
                        elif model_type_key == "RecurrentPPO" or model_type_key == "PPO": # Fallback PPO to RecurrentPPOBacktester for now
                            if model_type_key == "PPO":
                                st.warning("Generic PPO model type inferred, using RecurrentPPOBacktester. Create a specific PPOBacktester if needed.")
                            backtester = RecurrentPPOBacktester(
                                model_path=model_file_path, env_stats_path=env_stats,
                                config=st.session_state.config, ticker_list=selected_tickers
                            )
                        else:
                            st.error(f"Could not determine appropriate backtester for model: {model_info['name']}. Supported keywords: TD3, RecurrentPPO, PPO.")
                            st.stop()

                        if backtester:
                            backtester_output_dir_for_run = backtester.output_dir
                            metrics, results_df = backtester.run_full_backtest(backtest_data_source=processed_data_csv_path)

                            st.session_state.backtest_metrics_dict = metrics
                            st.session_state.backtest_results_df = results_df
                            st.success("Backtest completed!")
                            if backtester_output_dir_for_run:
                                st.info(f"Backtest artifacts (results CSV, plots, metrics txt) saved in: {backtester_output_dir_for_run}")
                        else:
                            st.error("Backtester could not be initialized.")
                    except Exception as e:
                        st.error(f"Error during backtest execution: {e}")
                        st.exception(e)

    # Display results if available in session state
    if st.session_state.backtest_metrics_dict and st.session_state.backtest_results_df is not None:
        st.subheader("Backtest Performance Metrics")
        metrics_to_display = st.session_state.backtest_metrics_dict

        num_metrics = len(metrics_to_display)
        max_cols = 4
        num_rows = (num_metrics + max_cols - 1) // max_cols
        metric_items = list(metrics_to_display.items())
        item_idx = 0
        for _ in range(num_rows):
            cols = st.columns(min(num_metrics - item_idx, max_cols))
            for col_idx in range(len(cols)):
                if item_idx < num_metrics:
                    key, value = metric_items[item_idx]
                    cols[col_idx].metric(label=str(key), value=str(value)) # Ensure value is string
                    item_idx += 1

        st.subheader("Performance Plots (from DataFrame)")
        results_df_for_plot = st.session_state.backtest_results_df.copy()

        # Ensure index is DatetimeIndex for plotting
        # BaseBacktester sets 'current_date' as index and sorts it.
        if not isinstance(results_df_for_plot.index, pd.DatetimeIndex):
            if 'current_date' in results_df_for_plot.columns: # If it was reset
                 results_df_for_plot['current_date'] = pd.to_datetime(results_df_for_plot['current_date'])
                 results_df_for_plot.set_index('current_date', inplace=True)
            elif results_df_for_plot.index.name == 'Date': # if it was reset and index became 'Date'
                 results_df_for_plot.index = pd.to_datetime(results_df_for_plot.index)
            # else: Fallback, index might be non-datetime, plots might look odd.

        if 'portfolio_value' in results_df_for_plot.columns:
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(x=results_df_for_plot.index, y=results_df_for_plot['portfolio_value'],
                                        mode='lines', name='Portfolio Value'))
            fig_pv.update_layout(title_text="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Portfolio Value ($)", height=500)
            st.plotly_chart(fig_pv, use_container_width=True)
        else:
            st.warning("Portfolio value data not found in results for plotting.")

        # BaseBacktester's _calculate_and_log_metrics adds 'drawdown_pct' and 'daily_returns'
        if 'drawdown_pct' in results_df_for_plot.columns:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=results_df_for_plot.index, y=-results_df_for_plot['drawdown_pct'],
                                         mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red')))
            fig_dd.update_layout(title_text="Drawdown Percentage Over Time", xaxis_title="Date", yaxis_title="Drawdown (%)", height=400)
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.warning("Drawdown percentage data not found for plotting.")

        if 'daily_returns' in results_df_for_plot.columns and not results_df_for_plot['daily_returns'].empty:
            fig_returns_dist = px.histogram(x=results_df_for_plot['daily_returns'] * 100, nbins=50, title="Distribution of Daily Returns")
            fig_returns_dist.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig_returns_dist, use_container_width=True)
        else:
            st.warning("Daily returns data not found or empty for plotting distribution.")

with tab_model_info:
    st.header("Model and Environment Information")
    if st.session_state.selected_model_info:
            max_cols = 4
            num_rows = (num_metrics + max_cols - 1) // max_cols
            metric_items = list(valid_metrics.items())
            item_idx = 0
            for _ in range(num_rows):
                cols = st.columns(min(num_metrics - item_idx, max_cols))
                for col_idx in range(len(cols)):
                    if item_idx < num_metrics:
                        key, value = metric_items[item_idx]
                        # Format certain metrics like Sharpe, Calmar as float with 2-3 decimal places
                        if isinstance(value, float):
                            value_str = f"{value:.3f}"
                        else:
                            value_str = str(value)
                        cols[col_idx].metric(label=key, value=value_str)
                        item_idx += 1

            st.subheader("Performance Plots")
            # ... (plotting logic remains similar) ...
            x_axis_data = portfolio_df_series.index
            if st.session_state.processed_data_df is not None and \
               len(st.session_state.processed_data_df.index) == len(portfolio_df_series): # Ensure alignment
                # Try to use the Date index from the processed data if available and aligned
                x_axis_data = st.session_state.processed_data_df.index
                x_axis_title = "Date"
            else: # Fallback to time steps if not aligned or data_df not available
                x_axis_data = pd.RangeIndex(start=0, stop=len(portfolio_df_series))
                x_axis_title = "Time Steps"

            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(x=x_axis_data, y=portfolio_df_series.values,
                                    mode='lines', name='Portfolio Value'))
            fig_pv.update_layout(title_text="Portfolio Value Over Time", xaxis_title=x_axis_title, yaxis_title="Portfolio Value ($)", height=500)
            st.plotly_chart(fig_pv, use_container_width=True)

            if daily_returns and len(daily_returns) > 0:
                fig_returns_dist = px.histogram(x=daily_returns, nbins=50, title="Distribution of Daily Returns")
                fig_returns_dist.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency", height=400)
                st.plotly_chart(fig_returns_dist, use_container_width=True)
            else:
                st.warning("No daily returns data to display histogram.")


with tab_model_info:
    st.header("Model and Environment Information")
    if st.session_state.selected_model_info:
        st.subheader("Selected Model Details")
        m_info = st.session_state.selected_model_info
        st.text(f"Name: {m_info['name']}")
        st.text(f"Full Path: {m_info['model_path']}") # Changed label for clarity
        st.text(f"VecNormalize Stats Path: {m_info['stats_path'] if m_info['stats_path'] else 'Not found'}")

        # Infer algo type (similar logic as in load_streamlit_model_and_env)
        model_name_lower = os.path.basename(m_info['model_path']).lower()
        parent_dir_name = os.path.basename(os.path.dirname(m_info['model_path'])).lower()
        algo_type = "Unknown"
        for hint in [model_name_lower, parent_dir_name]:
            if "ppo" in hint: algo_type = "PPO"; break
            elif "a2c" in hint: algo_type = "A2C"; break
            elif "td3" in hint: algo_type = "TD3"; break
            elif "ddpg" in hint: algo_type = "DDPG"; break
        st.text(f"Inferred Algorithm Type: {algo_type}")

    st.subheader("Configuration Overview (from shared_config.yaml)")
    # Display some key configurations, be careful not to show sensitive data if any
    if CONFIG:
        st.text(f"Project Name: {CONFIG.get('project_name', 'N/A')}")
        st.text(f"Default Tickers: {CONFIG.get('default_ticker_list', 'N/A')}")
        st.text(f"Technical Indicators: {len(CONFIG.get('tech_indicator_list', []))} configured")
        st.text(f"Log Directory: {CONFIG.get('log_dir', 'N/A')}")
        st.text(f"Model Directory: {CONFIG.get('model_dir', 'N/A')}")
        # Optionally, allow expanding to see more config details
        with st.expander("View Full Loaded Configuration (excluding sensitive if any added later)"):
            # Filter out potentially large or complex dicts for cleaner display initially
            simple_config_view = {k: v for k, v in CONFIG.items() if not isinstance(v, dict) or len(str(v)) < 200}
            st.json(simple_config_view, expanded=False)


    st.subheader("Environment Description (MultiStockTradingEnv)")
    st.markdown(MultiStockTradingEnv.__doc__ if MultiStockTradingEnv.__doc__ else "No detailed docstring found in MultiStockTradingEnv.")
    # Add more details if needed, or pull from a dedicated env description in config

st.sidebar.info("Refresh the page if you encounter issues after changing model types or parameters.")

# Removed ensure_dummy_dpp_components and its call
