import pandas as pd
import os
import logging

class StockDataPostProcessor:
    def __init__(self, log_file='postprocessing.log'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        self.logger.info("StockDataPostProcessor initialized.")
    
    def _handle_missing_values(self, df):
        df = df.ffill().bfill()
        
        threshold_col = len(df) * 0.5
        df = df.dropna(thresh=threshold_col, axis=1)
        
        threshold_row = len(df.columns) * 0.2
        df = df.dropna(thresh=threshold_row, axis=0)
        
        self.logger.info("Missing values handled successfully.")
        return df
    
    def _remove_outliers(self, df):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numerical_cols:
            if col != 'VM_t':  # Skip volatility metric column
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        self.logger.info("Outliers removed successfully.")
        return df
    
    def _cap_volatility_metrics(self, df):
        if 'VM_t' in df.columns:
            lower_cap = df['VM_t'].quantile(0.01)
            upper_cap = df['VM_t'].quantile(0.99)
            df['VM_t'] = df['VM_t'].clip(lower=lower_cap, upper=upper_cap)
        
        self.logger.info("Volatility metrics capped successfully.")
        return df
    
    def run_postprocess(self, input_dir, output_dir):
        self.logger.info("Starting post-processing pipeline...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
        
        for filename in os.listdir(input_dir):
            if not filename.endswith('.csv'):
                self.logger.warning(f"Skipping non-CSV file: {filename}")
                continue
            
            try:
                self.logger.info(f"Post-processing {filename}")
                file_path = os.path.join(input_dir, filename)
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                
                df = self._handle_missing_values(df)
                df = self._remove_outliers(df)
                df = self._cap_volatility_metrics(df)
                
                output_file = os.path.join(output_dir, f"post_processed_{filename}")
                df.to_csv(output_file)
                
                self.logger.info(f"Saved post-processed data to {output_file}")
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")

