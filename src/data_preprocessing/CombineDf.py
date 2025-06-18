import os
import pandas as pd

class DataMerger:
    def __init__(self, input_dir, macro_file, output_file):
        self.input_dir = input_dir
        self.macro_file = macro_file
        self.output_file = output_file
    
    def merge_data(self):
        # Load macroeconomic data
        macro_path = os.path.join(self.input_dir, self.macro_file)
        try:
            macro_df = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
            print("Loaded macro data")
        except FileNotFoundError:
            print(f"Macro file {self.macro_file} not found")
            raise
        except Exception as e:
            print(f"Error loading macro data: {e}")
            raise

        # Load all stock data files
        stock_files = [f for f in os.listdir(self.input_dir) if f.startswith('post_processed_') and f.endswith('.csv')]
        if not stock_files:
            print("No stock data files found")
            return

        # Initialize combined DataFrame with macro data
        combined_df = macro_df.reset_index()

        for stock_idx, file_name in enumerate(stock_files):
            file_path = os.path.join(self.input_dir, file_name)
            try:
                stock_df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date').reset_index()
                print(f"Processing {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue

            # Rename all columns (except Date) with stock index suffix
            non_date_columns = [col for col in stock_df.columns if col != 'Date']
            rename_mapping = {col: f"{col.lower()}_{stock_idx}" for col in non_date_columns}
            stock_df.rename(columns=rename_mapping, inplace=True)

            # Merge with combined dataframe
            combined_df = pd.merge(combined_df, stock_df, on='Date', how='outer', validate='1:1')
            print(f"Merged {file_name} into combined data")

        # Handle missing values: forward fill and drop any remaining NaN rows
        combined_df.ffill(inplace=True)
        combined_df.dropna(inplace=True)

        # Save the merged dataframe
        combined_df.to_csv(self.output_file, index=False)
        print(f"Successfully saved merged data to {self.output_file}")

    def run_combine_data(self):
        self.merge_data()
        return self.output_file

