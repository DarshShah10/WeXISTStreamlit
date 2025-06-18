import pandas as pd
import yfinance as yf
import logging

class MacroDataDownloader:
    def __init__(self, start_date="2017-01-01", end_date="2022-01-01"):
        
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    def _download_data(self, ticker, column_name):
        try:
            self.logger.info(f"Downloading {column_name} data...")
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df = df[['Close']].rename(columns={'Close': column_name})
            return df
        except Exception as e:
            self.logger.error(f"Failed to download {column_name} data: {e}")
            return pd.DataFrame()
    
    def run_download_macro(self, output_file):
        try:
            self.logger.info("Starting macroeconomic data download and preprocessing.")
            
            df_snp = self._download_data("^GSPC", "snp500")
            df_gold = self._download_data("GC=F", "gold_price")
            df_fed = self._download_data("^IRX", "interest_rate")
            
            df_merged = df_snp.join([df_gold, df_fed], how="outer")
            df_merged = df_merged.dropna(how="all")  # Remove rows with all missing values
            
            df_merged.to_csv(output_file, index=True)
            self.logger.info(f"Preprocessed macro data saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error during macro data processing: {e}")
