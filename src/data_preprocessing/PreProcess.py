import os
import shutil
class StockDataPreprocessor:
    def run_preprocessing(self, stock_data_dir, preprocessed_data_dir):
        print(f"[Dummy PreProcess] Called with stock_data_dir: {stock_data_dir}, preprocessed_data_dir: {preprocessed_data_dir}")
        os.makedirs(preprocessed_data_dir, exist_ok=True)
        # Simulate copying or processing files
        if os.path.exists(stock_data_dir):
            for item in os.listdir(stock_data_dir):
                s = os.path.join(stock_data_dir, item)
                d = os.path.join(preprocessed_data_dir, item)
                if os.path.isfile(s): shutil.copy2(s, d)
        print(f"[Dummy PreProcess] Copied files to {preprocessed_data_dir}")
