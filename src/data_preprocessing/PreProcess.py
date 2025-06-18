import pandas as pd
import os
import logging

class StockDataPreprocessor:
    
    def __init__(self, log_file='preprocessing.log'):
        self.threshold_column = 0.5  # 50% threshold for column dropping
        self.threshold_row = 0.2     # 20% threshold for row dropping
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        self.logger.info("StockDataPreprocessor initialized.")

    def _validate_file(self, file_path):
        try:
            df = pd.read_csv(file_path, skiprows=3)
            self.logger.info(f"Successfully read file: {file_path}")
        except pd.errors.ParserError:
            self.logger.error(f"Error reading file: {file_path}")
            return False
        return True

    def _process_columns(self, df):
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'EPS', 'PE_Ratio', 'Volatility_30D']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return None

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'EPS', 'PE_Ratio', 'Volatility_30D']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.logger.info("Columns processed successfully.")
        return df

    def _handle_missing_values(self, df):
        df = df.ffill().bfill()
        
        threshold_col = len(df) * self.threshold_column
        df = df.dropna(thresh=threshold_col, axis=1)
        
        threshold_row = len(df.columns) * self.threshold_row
        df = df.dropna(thresh=threshold_row, axis=0)
        
        self.logger.info("Missing values handled successfully.")
        return df

    def run_preprocessing(self, input_dir, output_dir):
       
        self.logger.info("Starting preprocessing pipeline...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")

        for filename in os.listdir(input_dir):
            if not filename.endswith('.csv'):
                self.logger.warning(f"Skipping non-CSV file: {filename}")
                continue

            try:
                self.logger.info(f"Processing file: {filename}")
                file_path = os.path.join(input_dir, filename)

                if not self._validate_file(file_path):
                    continue

                df = pd.read_csv(file_path, skiprows=3, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'EPS', 'PE_Ratio', 'Volatility_30D'])
                
                df = self._process_columns(df)
                if df is None:
                    self.logger.error(f"Column validation failed for {filename}")
                    continue

                df = self._handle_missing_values(df)
                output_file = os.path.join(output_dir, f"preprocessed_{filename}")
                df.to_csv(output_file)
                
                self.logger.info(f"Saved processed data to {output_file}")
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")

