import pandas as pd
import logging
import os

class MacroDataProcessor:
    def __init__(self, file_path="macro_data.csv", output_folder="post_processed_data"):
     
        self.file_path = file_path
        self.output_folder = output_folder
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def run_macro_postProcess(self):
        try:
            self.logger.info("Starting macro data post-processing.")
            
            # Load the data
            df = pd.read_csv(self.file_path, skiprows=3, names=['Date', 'snp500', 'gold_price', 'interest_rate'])
            
            # Validate required columns
            required_columns = ['Date', 'snp500', 'gold_price', 'interest_rate']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Missing required columns: {missing}")
                raise ValueError(f"Missing required columns: {missing}")
            
            # Convert date column and set as index
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df.set_index('Date', inplace=True)
            
            # Convert numeric columns
            numeric_columns = ['snp500', 'gold_price', 'interest_rate']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            df = df.ffill().bfill()
            
            # Drop columns/rows with excessive missing values
            threshold = len(df) * 0.5
            df = df.dropna(thresh=threshold, axis=1)
            row_threshold = len(df.columns) * 0.2
            df = df.dropna(thresh=row_threshold, axis=0)
            
            # Ensure the folder exists
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Save the preprocessed data
            output_file = os.path.join(self.output_folder, "macro.csv")
            df.to_csv(output_file, index=True)
            
            self.logger.info(f"Preprocessed data saved to {output_file}")
            print(df)
        except Exception as e:
            self.logger.error(f"Error during macro data processing: {e}")

