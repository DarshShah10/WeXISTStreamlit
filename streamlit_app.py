import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px # Added for Plotly plots
import plotly.graph_objects as go # Added for Plotly plots

# Stable Baselines3 components
from stable_baselines3 import PPO, A2C, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Project-specific imports
from src.data_preprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from src.envs.trading_env import MultiStockTradingEnv

# Import functions from train.py and test.py carefully
from train import (
    create_env,
    run_training_programmatically, # Added for model training tab
    DEFAULT_TICKER_LIST as TRAIN_DEFAULT_TICKERS,
    INITIAL_AMOUNT as TRAIN_INITIAL_AMOUNT,
    BUY_COST_PCT as TRAIN_BUY_COST_PCT,
    SELL_COST_PCT as TRAIN_SELL_COST_PCT,
    HMAX_PER_STOCK as TRAIN_HMAX_PER_STOCK,
    REWARD_SCALING as TRAIN_REWARD_SCALING,
    TECH_INDICATOR_LIST as TRAIN_TECH_INDICATOR_LIST,
    LOOKBACK_WINDOW as TRAIN_LOOKBACK_WINDOW,
    MAX_STEPS_PER_EPISODE as TRAIN_MAX_STEPS_PER_EPISODE
)
from test import calculate_metrics, plot_performance, run_evaluation

MODEL_SAVE_DIR = "models/"
DATA_DIR = "data/"
APP_TEMP_DATA_DIR = os.path.join(DATA_DIR, "app_temp")

@st.cache_data
def list_streamlit_models(model_dir):
    models = []
    zip_files = glob.glob(os.path.join(model_dir, "*.zip"))
    for model_path in zip_files:
        stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
        models.append({
            "name": os.path.basename(model_path),
            "model_path": model_path,
            "stats_path": stats_path if os.path.exists(stats_path) else None
        })

    best_model_dirs = glob.glob(os.path.join(model_dir, "best_model", "*"))
    for best_model_parent_dir in best_model_dirs:
        if os.path.isdir(best_model_parent_dir):
            best_model_path = os.path.join(best_model_parent_dir, "best_model.zip")
            best_model_stats_path = os.path.join(best_model_parent_dir, "vecnormalize.pkl")
            if os.path.exists(best_model_path):
                models.append({
                    "name": os.path.relpath(best_model_path, model_dir),
                    "model_path": best_model_path,
                    "stats_path": best_model_stats_path if os.path.exists(best_model_stats_path) else None
                })
    return models

def load_streamlit_model_and_env(selected_model_info, data_df, num_stocks_env):
    model_path = selected_model_info["model_path"]
    stats_path = selected_model_info["stats_path"]

    st.write(f"Loading Model: {model_path}")

    model_name_lower = os.path.basename(model_path).lower()
    AlgoClass = None
    if "ppo" in model_name_lower: AlgoClass = PPO
    elif "a2c" in model_name_lower: AlgoClass = A2C
    elif "td3" in model_name_lower: AlgoClass = TD3
    elif "ddpg" in model_name_lower: AlgoClass = DDPG
    elif "best_model.zip" in model_name_lower:
        parent_dir_name = os.path.basename(os.path.dirname(model_path)).lower()
        if "ppo" in parent_dir_name: AlgoClass = PPO
        elif "a2c" in parent_dir_name: AlgoClass = A2C
        elif "td3" in parent_dir_name: AlgoClass = TD3
        elif "ddpg" in parent_dir_name: AlgoClass = DDPG
        else: AlgoClass = PPO # Default if parent dir doesn't hint
    else:
        st.warning(f"Could not infer algorithm type from model name: {model_name_lower}. Attempting PPO.")
        AlgoClass = PPO

    import train # Import train module to set its global variables
    train.NUM_STOCKS = num_stocks_env
    train.INITIAL_AMOUNT = TRAIN_INITIAL_AMOUNT
    # Handle cases where TRAIN_BUY_COST_PCT might be a list of one item or need to be a list of num_stocks_env items
    train.BUY_COST_PCT = [TRAIN_BUY_COST_PCT[0]] * num_stocks_env if isinstance(TRAIN_BUY_COST_PCT, list) and len(TRAIN_BUY_COST_PCT) == 1 else [0.001] * num_stocks_env
    train.SELL_COST_PCT = [TRAIN_SELL_COST_PCT[0]] * num_stocks_env if isinstance(TRAIN_SELL_COST_PCT, list) and len(TRAIN_SELL_COST_PCT) == 1 else [0.001] * num_stocks_env
    train.HMAX_PER_STOCK = [TRAIN_HMAX_PER_STOCK[0]] * num_stocks_env if isinstance(TRAIN_HMAX_PER_STOCK, list) and len(TRAIN_HMAX_PER_STOCK) == 1 else [1000] * num_stocks_env
    train.REWARD_SCALING = TRAIN_REWARD_SCALING
    train.TECH_INDICATOR_LIST = TRAIN_TECH_INDICATOR_LIST
    train.LOOKBACK_WINDOW = TRAIN_LOOKBACK_WINDOW
    train.MAX_STEPS_PER_EPISODE = len(data_df) - 1 if not data_df.empty else TRAIN_MAX_STEPS_PER_EPISODE # Use data length for episode if possible

    raw_env = create_env(df=data_df, num_stocks_env=num_stocks_env, training=False)
    vec_env = DummyVecEnv([lambda: raw_env])

    if stats_path and os.path.exists(stats_path):
        st.write(f"Loading VecNormalize stats from: {stats_path}")
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        st.warning("No VecNormalize stats found. Using unnormalized environment.")

    model = AlgoClass.load(model_path, env=vec_env)
    st.success("Model loaded successfully.")
    return model, vec_env

st.set_page_config(page_title="RL Trading Dashboard", layout="wide", page_icon="📈")
st.title("🤖 Reinforcement Learning Trading Dashboard")

if "selected_model_info" not in st.session_state:
    st.session_state.selected_model_info = None
if "model" not in st.session_state:
    st.session_state.model = None
if "vec_env" not in st.session_state:
    st.session_state.vec_env = None
if "processed_data_df" not in st.session_state:
    st.session_state.processed_data_df = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "processed_train_data_path" not in st.session_state: # New session state for data processing tab
    st.session_state.processed_train_data_path = None

st.sidebar.header("⚙️ Configuration")
available_models = list_streamlit_models(MODEL_SAVE_DIR)

if not available_models:
    st.sidebar.warning("No trained models found in the 'models' directory.")
else:
    model_names = [m["name"] for m in available_models]
    selected_model_name = st.sidebar.selectbox(
        "Select Trained Model",
        options=model_names,
        index=0 if model_names else -1, # Handle empty model_names
        key="sb_select_model_name"
    )
    if selected_model_name: # Ensure a model is actually selected
        st.session_state.selected_model_info = next((m for m in available_models if m["name"] == selected_model_name), None)

st.sidebar.subheader("Data for Backtest/Inference")
user_tickers_str = st.sidebar.text_input(
    "Enter Stock Tickers (comma-separated)",
    value=", ".join(TRAIN_DEFAULT_TICKERS[:3]) if TRAIN_DEFAULT_TICKERS else "AAPL,MSFT,GOOG"
)
selected_tickers = [ticker.strip().upper() for ticker in user_tickers_str.split(",") if ticker.strip()]

default_start = datetime.now() - timedelta(days=3*365)
default_end = datetime.now() - timedelta(days=1)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=default_start, key="sb_start_date")
with col2:
    end_date = st.date_input("End Date", value=default_end, key="sb_end_date")

# Main application tabs
tab_data_processing, tab_model_training, tab_inference_backtesting, tab_model_info = st.tabs([
    "🛠️ Data Processing",
    "🧠 Model Training",
    "📊 Inference & Backtesting",
    "ℹ️ Model & Environment Info"
])

with tab_data_processing:
    st.header("Data Processing Pipeline")
    st.markdown("""
    Use this section to process historical market data for a list of stock tickers.
    The processed data will be saved as a CSV file and can be used for training new models
    or other analyses.
    """)

    dp_user_tickers_str = st.text_input(
        "Enter Stock Tickers for Processing (comma-separated)",
        value=", ".join(TRAIN_DEFAULT_TICKERS[:5]) if TRAIN_DEFAULT_TICKERS else "RELIANCE.NS,TCS.NS,INFY.NS",
        key="dp_tickers"
    )
    dp_selected_tickers = [ticker.strip().upper() for ticker in dp_user_tickers_str.split(",") if ticker.strip()]

    dp_col1, dp_col2 = st.columns(2)
    with dp_col1:
        dp_start_date = st.date_input(
            "Start Date for Processing",
            value=datetime.now() - timedelta(days=5*365), # Longer default for training data
            key="dp_start_date"
        )
    with dp_col2:
        dp_end_date = st.date_input(
            "End Date for Processing",
            value=datetime.now() - timedelta(days=1),
            key="dp_end_date"
        )

    if st.button("⚙️ Process Market Data", key="btn_process_market_data"):
        if not dp_selected_tickers:
            st.warning("Please enter at least one stock ticker for processing.")
        elif dp_start_date >= dp_end_date:
            st.error("Error: Start date must be before end date for processing.")
        else:
            with st.spinner(f"Processing data for: {', '.join(dp_selected_tickers)}..."):
                try:
                    os.makedirs(APP_TEMP_DATA_DIR, exist_ok=True)
                    # Ensure tickers in filename are sorted for consistency
                    sorted_dp_tickers_str = "_".join(sorted(dp_selected_tickers))
                    output_filename = f"processed_training_data_{sorted_dp_tickers_str}_{dp_start_date.strftime('%Y%m%d')}_{dp_end_date.strftime('%Y%m%d')}.csv"
                    output_csv_path = os.path.join(APP_TEMP_DATA_DIR, output_filename)

                    pipeline = StockPreProcessPipeline()
                    processed_path = pipeline.run_data_pipeline(
                        dp_selected_tickers,
                        dp_start_date.strftime('%Y-%m-%d'),
                        dp_end_date.strftime('%Y-%m-%d'),
                        output_csv_path
                    )
                    st.session_state.processed_train_data_path = processed_path
                    st.success(f"Market data processed successfully for {len(dp_selected_tickers)} tickers!")
                    st.info(f"Processed data saved to: {processed_path}")

                    # Optionally display a sample of the processed data
                    # df_sample = pd.read_csv(processed_path, nrows=5)
                    # st.dataframe(df_sample)

                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    st.exception(e)

    if st.session_state.processed_train_data_path:
        st.markdown(f"**Last processed training data path:** `{st.session_state.processed_train_data_path}`")
        # Add a button to download this data?
        # with open(st.session_state.processed_train_data_path, "rb") as fp:
        #     st.download_button(
        #         label="Download Processed Data (CSV)",
        #         data=fp,
        #         file_name=os.path.basename(st.session_state.processed_train_data_path),
        #         mime="text/csv"
        #     )

with tab_model_training:
    st.header("Train a New RL Model")
    st.markdown("""
    Configure the parameters below and start the training process.
    The trained model will be saved and can be selected from the sidebar for inference.
    """)

    train_algo_options = ["PPO", "A2C", "TD3", "DDPG", "RecurrentPPO"]
    train_selected_algo = st.selectbox(
        "Select Algorithm for Training",
        options=train_algo_options,
        index=0,
        key="train_algo_select"
    )

    train_tickers_str = st.text_input(
        "Stock Tickers for Training (comma-separated)",
        value=", ".join(TRAIN_DEFAULT_TICKERS[:5]) if TRAIN_DEFAULT_TICKERS else "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS",
        key="train_tickers"
    )
    train_selected_tickers = [ticker.strip().upper() for ticker in train_tickers_str.split(",") if ticker.strip()]

    train_col1, train_col2, train_col3 = st.columns(3)
    with train_col1:
        train_start_date = st.date_input(
            "Training Start Date",
            value=datetime.now() - timedelta(days=4*365), # Default: 4 years back
            key="train_start_date"
        )
    with train_col2:
        train_end_date = st.date_input(
            "Training End Date (for train data)",
            value=datetime.now() - timedelta(days=1*365), # Default: 1 year back (this is end of training period)
            key="train_end_date"
        )
    with train_col3:
        train_test_end_date = st.date_input(
            "Test Data End Date (for eval during training)",
            value=datetime.now() - timedelta(days=1), # Default: yesterday (this is end of test/eval period)
            key="train_test_end_date"
        )

    train_total_timesteps = st.number_input(
        "Total Training Timesteps",
        min_value=1000,
        value=50000, # Default from train.py
        step=1000,
        key="train_timesteps",
        help="Number of steps the agent interacts with the environment during training."
    )

    train_model_tag = st.text_input(
        "Model Name Tag (Optional)",
        key="train_model_tag",
        placeholder="e.g., my_ppo_run_1",
        help="A custom tag to include in the saved model's filename."
    )

    if st.button("🚀 Start Training", key="btn_start_training"):
        if not train_selected_tickers:
            st.warning("Please enter at least one stock ticker for training.")
        elif not train_selected_algo:
            st.warning("Please select an algorithm.")
        elif train_start_date >= train_end_date:
            st.error("Error: Training Start Date must be before Training End Date.")
        elif train_end_date >= train_test_end_date:
            st.error("Error: Training End Date must be before Test Data End Date.")
        else:
            st.info(f"Starting training for {train_selected_algo} with tickers: {', '.join(train_selected_tickers)}.")
            st.info(f"Training data: {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
            st.info(f"Evaluation data (during training): {train_end_date.strftime('%Y-%m-%d')} to {train_test_end_date.strftime('%Y-%m-%d')}")
            st.info(f"Total timesteps: {train_total_timesteps}. Model tag: '{train_model_tag}'.")

            with st.spinner(f"Training {train_selected_algo} model... This may take a while. Check your terminal/console for detailed logs."):
                try:
                    # Ensure dates are strings
                    start_date_str = train_start_date.strftime('%Y-%m-%d')
                    end_date_str = train_end_date.strftime('%Y-%m-%d')
                    test_end_date_str = train_test_end_date.strftime('%Y-%m-%d')

                    saved_model_path = run_training_programmatically(
                        selected_algo_name_param=train_selected_algo,
                        ticker_list_param=train_selected_tickers,
                        start_date_param=start_date_str,
                        end_date_param=end_date_str,
                        test_end_date_param=test_end_date_str,
                        total_timesteps_param=int(train_total_timesteps),
                        model_tag_param=train_model_tag
                    )
                    if saved_model_path:
                        st.success(f"Training finished! Model saved to: {saved_model_path}")
                        st.cache_data.clear() # Clear cache for list_streamlit_models
                        st.info("Model list updated. You might need to reselect the model in the sidebar if it was previously selected, or refresh the page.")
                    else:
                        st.error("Training failed or algorithm not found. Check console logs.")
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    st.exception(e)

with tab_inference_backtesting: # Renamed from tab_inference
    st.header("Run Inference / Backtest")

    if not st.session_state.selected_model_info:
        st.warning("Please select a model from the sidebar.")
    elif not selected_tickers:
        st.warning("Please enter at least one stock ticker in the sidebar.")
    elif start_date >= end_date:
        st.error("Error: Start date must be before end date.")
    else:
        st.write(f"**Selected Model**: {st.session_state.selected_model_info['name']}")
        st.write(f"**Tickers**: {', '.join(selected_tickers)}")
        st.write(f"**Date Range**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        if st.button("🚀 Process Data & Run Backtest", key="btn_run_backtest"):
            st.session_state.processed_data_df = None
            st.session_state.model = None
            st.session_state.vec_env = None
            st.session_state.backtest_results = None

            with st.spinner("Processing data..."):
                try:
                    os.makedirs(APP_TEMP_DATA_DIR, exist_ok=True)
                    # Ensure tickers in filename are sorted for consistency
                    sorted_selected_tickers_str = "_".join(sorted(selected_tickers))
                    output_filename = f"app_data_{sorted_selected_tickers_str}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    output_csv_path = os.path.join(APP_TEMP_DATA_DIR, output_filename)

                    pipeline = StockPreProcessPipeline()
                    processed_path = pipeline.run_data_pipeline(
                        selected_tickers,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        output_csv_path
                    )
                    data_df = pd.read_csv(processed_path)
                    data_df["Date"] = pd.to_datetime(data_df["Date"])
                    data_df.set_index("Date", inplace=True)
                    st.session_state.processed_data_df = data_df
                    st.success(f"Data processed successfully! Shape: {data_df.shape}")
                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    st.exception(e)

            if st.session_state.processed_data_df is not None and not st.session_state.processed_data_df.empty:
                with st.spinner("Loading model and preparing environment..."):
                    try:
                        num_processed_stocks = len(selected_tickers)
                        model, vec_env = load_streamlit_model_and_env(
                            st.session_state.selected_model_info,
                            st.session_state.processed_data_df,
                            num_processed_stocks
                        )
                        st.session_state.model = model
                        st.session_state.vec_env = vec_env
                    except Exception as e:
                        st.error(f"Error loading model or environment: {e}")
                        st.exception(e)

            if st.session_state.model and st.session_state.vec_env and st.session_state.processed_data_df is not None:
                with st.spinner("Running backtest..."):
                    try:
                        portfolio_values, daily_returns = run_evaluation(
                            st.session_state.model,
                            st.session_state.vec_env,
                            st.session_state.processed_data_df
                        )
                        st.session_state.backtest_results = (portfolio_values, daily_returns)
                        st.success("Backtest completed!")
                    except Exception as e:
                        st.error(f"Error during backtest execution: {e}")
                        st.exception(e)

    if st.session_state.backtest_results:
        st.subheader("Backtest Results")
        portfolio_values, daily_returns = st.session_state.backtest_results

        if not portfolio_values or len(portfolio_values) < 2:
            st.warning("Not enough data in portfolio values to display metrics.")
        else:
            # Use TRAIN_INITIAL_AMOUNT for consistency if run_evaluation doesn't provide it
            # This assumes run_evaluation's portfolio_values starts with this amount.
            metrics, portfolio_df_series = calculate_metrics(portfolio_values, daily_returns)

            # Display metrics in columns
            # Filter out any metrics that might be None or not suitable for st.metric
            valid_metrics = {k: v for k, v in metrics.items() if v is not None}

            num_metrics = len(valid_metrics)
            # Create more rows if too many metrics for one row of columns
            max_cols = 4
            num_rows = (num_metrics + max_cols - 1) // max_cols

            metric_items = list(valid_metrics.items())
            item_idx = 0
            for _ in range(num_rows):
                cols = st.columns(min(num_metrics - item_idx, max_cols))
                for col_idx in range(len(cols)):
                    if item_idx < num_metrics:
                        key, value = metric_items[item_idx]
                        cols[col_idx].metric(label=key, value=str(value))
                        item_idx += 1

            st.subheader("Performance Plots")

            # Portfolio Value Over Time
            x_axis_data = portfolio_df_series.index # Default to time steps
            if st.session_state.processed_data_df is not None and \
               len(st.session_state.processed_data_df.index) == len(portfolio_df_series):
                x_axis_data = st.session_state.processed_data_df.index
                x_axis_title = "Date"
            else:
                x_axis_title = "Time Steps"

            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(x=x_axis_data, y=portfolio_df_series.values,
                                    mode='lines', name='Portfolio Value'))
            fig_pv.update_layout(
                title_text="Portfolio Value Over Time",
                xaxis_title=x_axis_title,
                yaxis_title="Portfolio Value ($)",
                height=500
            )
            st.plotly_chart(fig_pv, use_container_width=True)

            # Distribution of Daily Returns
            if daily_returns: # Check if daily_returns is not empty
                fig_returns_dist = px.histogram(
                    x=daily_returns,
                    nbins=50,
                    title="Distribution of Daily Returns"
                )
                fig_returns_dist.update_layout(
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_returns_dist, use_container_width=True)
            else:
                st.warning("No daily returns data to display histogram.")

with tab_model_info:
    st.header("Model and Environment Information")
    if st.session_state.selected_model_info:
        st.subheader("Selected Model Details")
        m_info = st.session_state.selected_model_info
        st.text(f"Name: {m_info['name']}")
        st.text(f"Path: {m_info['model_path']}")
        st.text(f"VecNormalize Stats: {m_info['stats_path'] if m_info['stats_path'] else 'Not found'}")

        model_name_lower = os.path.basename(m_info['model_path']).lower()
        algo_type = "Unknown"
        if "ppo" in model_name_lower: algo_type = "PPO"
        elif "a2c" in model_name_lower: algo_type = "A2C"
        elif "td3" in model_name_lower: algo_type = "TD3"
        elif "ddpg" in model_name_lower: algo_type = "DDPG"
        elif "best_model.zip" in model_name_lower: # Check parent folder for algo type hint
             parent_dir_name = os.path.basename(os.path.dirname(m_info['model_path'])).lower()
             if "ppo" in parent_dir_name: algo_type = "PPO"
             elif "a2c" in parent_dir_name: algo_type = "A2C"
             elif "td3" in parent_dir_name: algo_type = "TD3"
             elif "ddpg" in parent_dir_name: algo_type = "DDPG"
        st.text(f"Inferred Algorithm Type: {algo_type}")

    st.subheader("Environment Description (MultiStockTradingEnv)")
    st.markdown(
        """
The `MultiStockTradingEnv` is a custom environment for training reinforcement learning agents
to trade multiple stocks.

- **Action Space**: Continuous values representing buy/sell/hold actions for each selected stock.
- **Observation Space**: Includes normalized market data (OHLCV), technical indicators,
  macro-economic data, current cash, and shares held for each stock.
- **Reward Function**: Aims to maximize portfolio value while considering factors like
  risk-adjusted returns and penalizing excessive drawdown.
- **Data Source**: Utilizes historical stock data fetched via yfinance (through the
  preprocessing pipeline) and relevant macroeconomic indicators.
        """
    )

st.sidebar.info("Refresh the page if you encounter issues after changing model types or parameters.")

# This function is mainly a placeholder if complex setup is needed before app runs.
# For now, dummy files are created by train.py's test run or should be part of repo.
@st.cache_resource
def ensure_dummy_dpp_components():
    # Example: ensure src/data_preprocessing/CollectData.py exists with minimal content
    # if not os.path.exists("src/data_preprocessing/CollectData.py"):
    #     # Create dummy file or raise error
    #     pass
    pass

ensure_dummy_dpp_components()
