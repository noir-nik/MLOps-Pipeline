from matplotlib import pyplot as plt
import seaborn as sns
from common.config_static import DATA_STATE_DIR, RAW_DATA_DIR, REPORTS_DIR
from common.logger import logger
from common.config import Config
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple
import os
import json

class DataCollector:
    """Handles data collection and stream simulation"""
    
    def __init__(self, config: Config, data_path: str = None):
        """Initialize data collector"""
        self.config = config
        self.data_path = data_path
        self.batch_size = self.config.get("data_collection", "batch_size")
        self.raw_data_path = RAW_DATA_DIR
        self.latest_batch = -1
        self.metadata = {
            "total_records": 0,
            "total_batches": 0,
            "last_update": None,
            "data_sources": [],
            "column_types": {},
        }
        self.metadata_path = os.path.join(RAW_DATA_DIR, "metadata.json")
        self.load_metadata()
        self.load_data_state()
    
    def set_latest_batch(self, batch_id: int):
        self.latest_batch = batch_id
        self.save_data_state()

    def is_data_available(self) -> bool:
        return self.metadata["total_batches"] > 0
    
    def load_metadata(self) -> bool:
        """Load metadata from file if exists"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as file:
                self.metadata = json.load(file)
            logger.info(f"Metadata loaded: {len(self.metadata['batches_info'])} batches recorded")
            return True
        return False
    
    def save_metadata(self):
        """Save metadata to file"""
        self.metadata["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metadata_path, 'w') as file:
            json.dump(self.metadata, file, indent=2)
        logger.info("Metadata saved")
    
    def collect_initial_data(self, file_path: str) -> pd.DataFrame:
        """Collect initial data from file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide CSV or Excel file.")

            # Sort the dataframe by the 'Date' column
            df = df.sort_values(by='Date')
            
            # Update metadata
            self.metadata["total_records"] = len(df)
            self.metadata["data_sources"].append(file_path)
            self.metadata["column_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            self.metadata["records"] = len(df)
            self.metadata["columns"] = len(df.columns)
            self.metadata["missing_values"] = df.isna().sum().to_dict()
            self.metadata["missing_percentage"] = (df.isna().sum() / len(df) * 100).to_dict()
            self.metadata["categorical_columns"] = list(df.select_dtypes(include=['object', 'category']).columns)
            self.metadata["numerical_columns"] = list(df.select_dtypes(include=['int64', 'float64']).columns)
            self.metadata["memory_usage"] = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            # Create simulated stream batches
            self.create_batches(df)
            self.save_metadata()
            
            return df
        except Exception as e:
            logger.error(f"Error collecting initial data: {e}")
            raise
    
    def create_batches(self, df: pd.DataFrame):
        """Create batches for simulated streaming"""
        total_rows = len(df)
        num_batches = total_rows // self.batch_size + (1 if total_rows % self.batch_size > 0 else 0)
        
        logger.info(f"Creating {num_batches} batches from {total_rows} records")
        
        batch_info = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_rows)
            
            batch_df = df.iloc[start_idx:end_idx].copy()
            batch_filepath = os.path.join(self.raw_data_path, f"batch_{i}.csv")
            batch_df.to_csv(batch_filepath, index=False)
            
            batch_info.append({
                "batch_id": i,
                "records": len(batch_df),
                "file_path": batch_filepath,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logger.debug(f"Batch {i} created with {len(batch_df)} records")
        
        self.metadata["batches_info"] = batch_info
        self.metadata["total_batches"] = num_batches
        self.save_metadata()
    
    def get_next_batch(self) -> Tuple[pd.DataFrame, int]:
        """Get next batch for processing"""
        self.set_latest_batch(self.latest_batch + 1)
        return self.get_batch(self.latest_batch), self.latest_batch

    def get_latest_batch(self) -> Tuple[pd.DataFrame, int]:
        return self.get_batch(self.latest_batch), self.latest_batch

    def get_batch(self, batch_id: int) -> Tuple[pd.DataFrame, int]:
        """Get next batch for processing"""

        if batch_id is None or batch_id >= self.metadata["total_batches"]:
            logger.info("No more batches available")
            return None, -1
        
        batch_info = self.metadata["batches_info"][batch_id]
        batch_path = batch_info["file_path"]
        
        try:
            batch_df = pd.read_csv(batch_path)
            logger.info(f"Loaded batch {batch_id} with {len(batch_df)} records")
        
            
            return batch_df
        except Exception as e:
            logger.error(f"Error loading batch {batch_id}: {e}")
            return self.get_next_batch()
    
    def save_data_state(self):
        """Save current_batch number to file"""
        data_state_path = os.path.join(DATA_STATE_DIR, "data_state.json")
        data_state = {
            "current_batch": self.latest_batch
        }
        with open(data_state_path, 'w') as file:
            json.dump(data_state, file, indent=2)
        logger.info(f"Data state saved: {data_state_path}")
    
    def load_data_state(self):
        """Load current_batch number from file"""
        data_state_path = os.path.join(DATA_STATE_DIR, "data_state.json")
        if os.path.exists(data_state_path):
            with open(data_state_path, 'r') as file:
                data_state = json.load(file)
            loaded = data_state["current_batch"]
            self.latest_batch = loaded if loaded >= -1 else -1
            logger.info(f"Data state loaded: current_batch={self.latest_batch}")
        else:
            logger.info("Data state not found")

    
    def reset_batch_counter(self):
        """Reset batch counter to start from beginning"""
        self.latest_batch = 0
        logger.info("Batch counter reset")
    
    def calculate_metadata(self, df: pd.DataFrame) -> Dict:
        """Calculate metadata for a dataframe"""
        metadata = {
            "records": len(df),
            "columns": len(df.columns),
            "missing_values": df.isna().sum().to_dict(),
            "missing_percentage": (df.isna().sum() / len(df) * 100).to_dict(),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "numerical_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        }
        return metadata
    
    def generate_eda_report(self, df: pd.DataFrame, batch_id: int):
        """Generate automatic EDA report"""
        report_path = os.path.join(REPORTS_DIR, f"eda_report_batch_{batch_id}.html")
        
        try:
            import pandas_profiling
            from pandas_profiling import ProfileReport
            
            profile = ProfileReport(df, title=f"Data Profiling Report - Batch {batch_id}")
            profile.to_file(report_path)
            logger.info(f"EDA report generated: {report_path}")
            return report_path
        except ImportError:
            logger.warning("pandas-profiling not installed. Skipping EDA report generation.")
            return None
        
        except Exception as e:
            logger.error(f"Error generating EDA report: {e}")
            return None