# Reinforcement Learning Stock Trading Project

This project implements a stock trading environment using OpenAI Gymnasium and trains reinforcement learning agents using Stable-Baselines3.

## Project Structure

```
project_root/
├── src/
│   ├── agents/            # RL agent setups (e.g., PPO, TD3 specific configs, not fully populated yet)
│   ├── envs/              # Custom Gym environments (MultiStockTradingEnv)
│   ├── rewards/           # Reward functions (simple_portfolio_value_change)
│   ├── utils/             # Logging, plotting, wrappers, helpers (wrappers.py exists)
│   ├── config/            # Config files (not populated yet, main.py has config vars)
│   └── data_preprocessing/ # Scripts for data collection and preprocessing (moved from old structure)
├── main.py                # Entrypoint for training the RL agent
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── trained_models/        # Directory where trained models will be saved by main.py
├── logs/                  # Directory for SB3 logs (TensorBoard, evaluation logs)
├── notebooks/             # Jupyter notebooks (e.g., for EDA, moved from old structure)
└── project_root/models/   # Original models directory (if any pre-trained models existed)
    └── models_root/       # Contains original model files like .pkl, .zip, .bin
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <project-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing

The environment expects preprocessed data. The scripts for this are currently located in `src/data_preprocessing/`.
You would typically run a pipeline (e.g., `src/data_preprocessing/DataPreprocessingPipeline.py` if adapted) to generate a CSV file that the trading environment can consume.

**Note:** The current `main.py` simulates data preprocessing by creating a dummy DataFrame. For actual use, replace this with your data loading and preprocessing logic. The dummy data expects columns like `open_0, high_0, ..., close_0, volume_0, eps_0, pe_ratio_0, volatility_30d_0, momentum_0, volume_trend_0, macd_0, rsi_30_0, ...` for each stock, plus `snp500, gold_price, interest_rate` for macro features.

### Training

To train an RL agent, run the `main.py` script:

```bash
python main.py
```

This script will:
1.  Load/simulate data.
2.  Initialize the `MultiStockTradingEnv`.
3.  Select an agent (default PPO, configurable in `main.py`).
4.  Train the agent using Stable-Baselines3.
5.  Save logs to the `./logs/` directory (viewable with TensorBoard).
6.  Save model checkpoints and the final model to the `./trained_models/` directory.

### Configuration

Key configurations (tickers, dates, environment parameters, agent settings) are located at the beginning of `main.py`. These can be modified directly or moved to a dedicated configuration file (e.g., in `src/config/`) for more complex setups.

## Environment Details (`src/envs/New_Trading_Env.py`)

*   **Action Space**: Continuous, `Box(low=-1.0, high=1.0, shape=(num_stocks,), dtype=np.float32)`. Each action element represents the percentage of resources to allocate to buying (if positive) or selling (if negative) a particular stock.
*   **Observation Space**: Continuous, includes normalized cash, normalized shares held, scaled price data (OHLCV), scaled technical indicators, and scaled macro-economic features.
*   **Reward Function**: Currently, a simple reward based on the change in portfolio value: `reward = current_portfolio_value - previous_portfolio_value`. This is defined in `src/rewards/reward_functions.py`.

## Key Architectural Changes

*   **Modular Structure**: Codebase organized into `src` subdirectories for better separation of concerns (agents, environments, rewards, utils, data_preprocessing, config).
*   **Gymnasium Compliance**: The custom environment `MultiStockTradingEnv` has been updated to comply with the Gymnasium API (formerly Gym API), including `reset` and `step` method signatures.
*   **Simplified Reward**: The complex, multi-parameter reward function has been replaced with a simpler, more direct reward based on portfolio value change. The old reward logic is commented out in the environment file for reference.
*   **Centralized Training Script**: `main.py` provides a clear entry point for training and basic evaluation, using Stable-Baselines3.
*   **Standardization**: Adherence to common practices for RL projects using SB3.

## Future Work / Improvements

*   **Implement Data Preprocessing Integration**: Connect the `StockPreProcessPipeline` (or equivalent) to `main.py` to use real, processed data.
*   **Configuration Management**: Move configurations from `main.py` to dedicated files (e.g., YAML or Python config files in `src/config/`).
*   **Advanced Evaluation**: Implement more robust backtesting and evaluation metrics.
*   **Hyperparameter Tuning**: Add scripts or notebooks for hyperparameter optimization (e.g., using Optuna).
*   **Agent Customization**: Further develop agent-specific configurations or custom policies in `src/agents/`.
*   **Full Data Pipeline**: Ensure the data preprocessing steps correctly generate all features expected by the environment, including the technical indicators listed in `TECH_INDICATOR_LIST`.
*   **Stop-Loss Logic**: Review and test the stop-loss mechanism in the environment.
