import yaml
import os

def load_config(config_path="config/shared_config.yaml"):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file,
                           relative to the project root.

    Returns:
        dict: A dictionary containing the configuration.
    """
    # Construct absolute path to the config file
    # Assuming this script is in src/utils, project root is two levels up.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    absolute_config_path = os.path.join(project_root, config_path)

    if not os.path.exists(absolute_config_path):
        raise FileNotFoundError(f"Configuration file not found at: {absolute_config_path}")

    with open(absolute_config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            if config is None: # Handle empty YAML file case
                config = {}
            config['PROJECT_ROOT'] = project_root
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    # Resolve relative paths in config to absolute paths from project_root
    # This makes paths usable directly from anywhere in the project
    paths_to_resolve = {
        "data_dir": [],
        "model_dir": [],
        "log_dir": [],
        "config_dir": [], # Though this one is usually relative to project root itself
        "fundamental_analysis_dir": [],
        "raw_stock_data_dir": ["data_dir"],
        "processed_stock_data_dir": ["data_dir"],
        "feature_engineered_data_dir": ["data_dir"],
        "final_combined_data_dir": ["data_dir"],
        "macro_data_filename": ["data_dir"],
        "processed_macro_filename": ["data_dir"],
        "xgboost_model_filename": ["model_dir"], # Or fundamental_analysis_dir, handle logic in usage
        "tensorboard_log_dir": ["log_dir"],
        "checkpoint_dir": ["model_dir"],
        "best_model_dir": ["model_dir"],
        "backtest_output_dir": ["log_dir"], # Or a dedicated output_dir
        "trading_metrics_filename": ["log_dir"] # Or a dedicated output_dir
    }

    for key, base_keys in paths_to_resolve.items():
        if key in config:
            path_parts = [project_root]
            for base_key in base_keys:
                if base_key in config:
                    path_parts.append(config[base_key])
                else:
                    # print(f"Warning: Base path key '{base_key}' not found for '{key}' in config. Assuming relative to project root.")
                    pass # Or handle as an error
            path_parts.append(config[key])
            config[key] = os.path.abspath(os.path.join(*path_parts))
        # else:
            # print(f"Warning: Path key '{key}' not found in config.")

    # For base directories, ensure they are absolute from project_root if not already
    base_dirs = ["data_dir", "model_dir", "log_dir", "config_dir", "fundamental_analysis_dir"]
    for base_dir_key in base_dirs:
        if base_dir_key in config and not os.path.isabs(config[base_dir_key]):
            config[base_dir_key] = os.path.abspath(os.path.join(project_root, config[base_dir_key]))

    # config['project_root'] = project_root # This line is now redundant as it's added earlier
    return config

if __name__ == '__main__':
    # Example usage:
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        # Print some resolved paths to test
        print(f"Project Root: {config.get('project_root')}")
        print(f"Data Directory: {config.get('data_dir')}")
        print(f"Raw Stock Data Directory: {config.get('raw_stock_data_dir')}")
        print(f"XGBoost Model Path: {config.get('xgboost_model_filename')}") # Example of a specific file
        print(f"Default Tickers: {config.get('default_ticker_list')}")
    except Exception as e:
        print(f"Error: {e}")
