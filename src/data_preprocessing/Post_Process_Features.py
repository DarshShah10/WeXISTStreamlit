import os
import shutil
class StockDataPostProcessor:
    def run_postprocess(self, feature_added_data_dir, post_processed_data_dir):
        print(f"[Dummy PostProcess] Called with feature_added_data_dir: {feature_added_data_dir}, post_processed_data_dir: {post_processed_data_dir}")
        os.makedirs(post_processed_data_dir, exist_ok=True)
        if os.path.exists(feature_added_data_dir):
            for item in os.listdir(feature_added_data_dir):
                s = os.path.join(feature_added_data_dir, item)
                d = os.path.join(post_processed_data_dir, item)
                if os.path.isfile(s): shutil.copy2(s, d) # Simulate processing
        print(f"[Dummy PostProcess] Processed files to {post_processed_data_dir}")
