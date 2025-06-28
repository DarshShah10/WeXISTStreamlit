# RL Trading Framework

## Overview

This project is a Python-based framework for developing and testing reinforcement learning (RL) agents for stock trading. It includes a modular data processing pipeline, support for various RL algorithms via Stable Baselines3, a centralized configuration system, a custom trading environment, a backtesting framework, and a Streamlit web application for easy interaction. A fundamental analysis module using XGBoost for stock scoring is also included.

## Features

*   **Modular Data Processing Pipeline**: Automated collection of stock and macroeconomic data, preprocessing, comprehensive feature engineering (technical indicators, fundamental features), and data merging.
*   **RL Algorithm Support**: Easily train agents using algorithms like PPO, A2C, TD3, DDPG, and RecurrentPPO from Stable Basolines3.
*   **Centralized Configuration**: All project parameters (paths, data settings, environment variables, model hyperparameters) are managed via `config/shared_config.yaml`.
*   **Custom Trading Environment**: `MultiStockTradingEnv` simulates trading multiple stocks with configurable costs, initial capital, and a sophisticated reward function.
*   **Backtesting Framework**: Evaluate trained agents with detailed performance metrics (Sharpe Ratio, Sortino Ratio, Max Drawdown, etc.) and performance plots.
*   **Streamlit Web Application**: User-friendly interface for:
    *   Processing market data.
    *   Training various RL models.
    *   Running backtests and visualizing performance.
    *   Viewing model and environment information.
*   **Fundamental Analysis Module**: Includes tools to train an XGBoost model based on stock fundamentals to generate predictive scores.

## Directory Structure

```
.
├── config/
│   └── shared_config.yaml        # Central configuration file
├── data/                         # Root for all data (see config for subdirectories)
│   ├── raw_stock_data/           # Raw downloaded stock CSVs
│   ├── processed_stock_data/     # Cleaned stock CSVs
│   ├── feature_engineered_data/  # Stock CSVs with added features
│   ├── post_processed_individual_stock_data/ # Final individual stock CSVs
│   ├── final_combined_data/      # Merged data for the RL environment
│   └── app_temp/                 # Temporary data for Streamlit app operations
│   # (macro_economic_data.csv and processed_macro_economic_data.csv are also in data/)
├── Fundamental/
│   ├── Fundamental.py            # Main script for training fundamental XGBoost model
│   └── test_fundamental.py       # Script to test/use the trained fundamental model
├── logs/
│   ├── backtest_results/         # Output from backtesting runs (metrics, plots, raw data)
│   └── tensorboard_logs/         # Logs for TensorBoard visualization during training
│   # (trading_environment_metrics.csv may also be here)
├── models/                       # Saved trained models
│   ├── checkpoints/              # Intermediate model checkpoints during training
│   └── best_model/               # Best models saved during evaluation callbacks
│   # (RL agent .zip files, VecNormalize .pkl files, and xgboost_fundamental_model.bin)
├── src/
│   ├── data_preprocessing/       # Modules for the data pipeline
│   │   ├── CollectData.py
│   │   ├── Download_Macro.py
│   │   ├── PreProcess.py
│   │   ├── PreProcessMacro.py
│   │   ├── FeatureEngineer.py
│   │   ├── Post_Process_Features.py
│   │   ├── CombineDf.py
│   │   └── DataPreprocessingPipeline.py # Orchestrator
│   ├── envs/                     # Trading environment and backtesting framework
│   │   ├── trading_env.py
│   │   ├── base_backtester.py
│   │   ├── backtest.py           # TD3Backtester
│   │   └── backtestppo.py        # RecurrentPPOBacktester
│   └── utils/                    # Utility functions
│       └── config_loader.py
├── REQUIREMENTS.txt              # Python dependencies (Note: Corrected filename)
├── README.md                     # This file
├── main.py                       # Example script (if any, for core logic tests - currently not primary focus)
├── streamlit_app.py              # Main Streamlit application script
├── train.py                      # Script for training RL models
└── test.py                       # Original test script (may need refactoring or removal)
```

*(Note: The `REQUIREMENTS.txt` was referred to as `requirements.txt` in the subtasks. Assuming standard naming.)*

## Setup and Installation

1.  **Python Version**: Python 3.9+ is recommended (as per common ML libraries). Check `config/shared_config.yaml` for a specific version if set.
2.  **Create Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r REQUIREMENTS.txt
    ```
4.  **Configuration**:
    *   Review and customize `config/shared_config.yaml`. This file controls:
        *   File paths for data, models, and logs.
        *   Default stock tickers and date ranges.
        *   Technical indicators and macroeconomic features to use.
        *   Trading environment parameters (initial capital, costs, reward function settings).
        *   Hyperparameters for RL model training.
        *   Fundamental analysis settings.
    *   Ensure paths like `data_dir`, `model_dir`, `log_dir` are appropriate for your system. The `PROJECT_ROOT` is determined automatically by `config_loader.py`.

## Usage

### 1. Data Processing

The data processing pipeline is typically run automatically as part of model training or backtesting if the required data files are not found. It sources its parameters (tickers, dates, features to generate, paths) from `config/shared_config.yaml`.

*   **Raw stock data** is saved in the directory specified by `raw_stock_data_dir` (e.g., `data/raw_stock_data/`).
*   **Macroeconomic data** is saved as configured by `macro_data_filename`.
*   Intermediate processed files are stored in their respective configured directories (e.g., `processed_stock_data/`, `feature_engineered_data/`).
*   The **final merged DataFrame** used by the RL environment is saved in `final_combined_data_dir` (e.g., `data/final_combined_data/`).

You can also trigger data processing via the "Data Processing" tab in the Streamlit application.

### 2. Training RL Agents

**Using `train.py` (Command Line)**:
1.  Navigate to the project root directory.
2.  Ensure your virtual environment is activated.
3.  Run the training script:
    ```bash
    python train.py
    ```
4.  You will be prompted to select an RL algorithm to train (e.g., PPO, TD3).
5.  The script uses parameters defined in `config/shared_config.yaml` for tickers, dates, environment settings, and algorithm-specific hyperparameters.
6.  Trained models, checkpoints, and VecNormalize statistics are saved in the `models/` directory (and its subdirectories `checkpoints/`, `best_model/`). TensorBoard logs are saved in `logs/tensorboard_logs/`.

**Using the Streamlit Application**:
1.  Launch the app (see section below).
2.  Navigate to the "Model Training" tab.
3.  Select the algorithm, specify tickers, date ranges, total timesteps, and an optional model tag.
4.  Click "Start Training". Progress can be monitored in the console where Streamlit is running.

### 3. Backtesting and Evaluation

**Using the Streamlit Application**:
1.  Launch the app.
2.  Navigate to the "Inference & Backtesting" tab.
3.  Select a trained model from the sidebar.
4.  Enter the stock tickers and date range for the backtest.
5.  Click "Process Data & Run Backtest".
    *   The app will first process the data for the selected parameters if a combined file doesn't already exist.
    *   It then runs the backtest using the selected model and the appropriate refactored backtester (`TD3Backtester` or `RecurrentPPOBacktester`).
6.  Results, including performance metrics (Sharpe Ratio, Max Drawdown, etc.) and plots (Portfolio Value, Drawdown, Returns Distribution), will be displayed in the app.
7.  Detailed results (raw simulation data CSV, metrics text file, plot images) are saved to a unique subdirectory within `logs/backtest_results/`.

### 4. Streamlit Application

1.  Ensure all dependencies are installed and `config/shared_config.yaml` is configured.
2.  Navigate to the project root directory.
3.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  The application provides tabs for:
    *   **Data Processing**: Manually trigger the data pipeline for specified tickers and dates.
    *   **Model Training**: Train new RL agents.
    *   **Inference & Backtesting**: Load trained models and evaluate their performance on historical data.
    *   **Model & Environment Info**: Displays details about the selected model and the trading environment.

### 5. Fundamental Analysis Module

This module trains a separate XGBoost model to score stocks based on their fundamental data.

**Training the Fundamental Model**:
1.  Run `Fundamental/Fundamental.py`:
    ```bash
    python Fundamental/Fundamental.py
    ```
2.  This script will:
    *   Fetch a list of NIFTY stocks (URL configurable in `shared_config.yaml`).
    *   Collect current fundamental data for these stocks using `yfinance`.
    *   Train an XGBoost model (hyperparameters and number of stocks for training are configurable).
    *   Save the trained model to the path specified in `config` (e.g., `models/xgboost_fundamental_model.bin`).
    *   Print and return a list of top-scoring stocks.

**Using the Trained Fundamental Model for Predictions**:
1.  Run `Fundamental/test_fundamental.py`:
    ```bash
    python Fundamental/test_fundamental.py
    ```
2.  This script will:
    *   Load the pre-trained XGBoost model from the configured path.
    *   Fetch current fundamental data for a list of stocks.
    *   Use the model to predict scores and display the top-ranking stocks.

## Troubleshooting

*   **Configuration Errors**: If the application fails to start or behaves unexpectedly, double-check `config/shared_config.yaml` for correct paths and parameter formats. Ensure `PROJECT_ROOT` is correctly determined by `src/utils/config_loader.py`.
*   **Data Download Issues**: Failures in `yfinance` or `pandas_datareader` might be due to network issues, delisted tickers, or API changes. Check console logs for specific error messages.
*   **Dependency Problems**: Ensure all packages in `REQUIREMENTS.txt` are installed correctly in your virtual environment.
*   **`FileNotFoundError` for models/data**: Verify that training/data processing has completed successfully and that the paths in `config/shared_config.yaml` point to the correct locations.

## Future Enhancements / TODOs

*   Refactor `test.py` to use the new `BaseBacktester` framework or remove it if functionality is fully covered by Streamlit app and individual backtester scripts.
*   Implement metadata saving alongside trained RL models (e.g., algorithm type, key parameters) to avoid relying on filename parsing for model loading.
*   More sophisticated error handling and user feedback in the Streamlit application.
*   Advanced options for feature selection and hyperparameter tuning.
*   Support for more data sources and financial instruments.
*   More detailed unit and integration tests.
