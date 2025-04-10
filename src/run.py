import argparse
from datetime import datetime
import json
import logging
import os
import sys
from typing import Dict
import numpy as np

import pandas as pd

from common.config import Config
from common.config_static import *
from common.logger import logger, init_logger
from common.numpy_encoder import NumpyEncoder
from data.data_collector import DataCollector
from data.data_preparer import DataPreparer
from data.data_source import download_kaggle_dataset
from model.model_trainer import ModelTrainer
from model.model_visualizer import ModelVisualizer


def create_dirs():
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument("-mode", type=str, required=True, help="Mode to run the script: inference, update, or summary")
    parser.add_argument("-file", type=str, help="Path to the file for inference mode")
    args = parser.parse_args()
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        return None
    if args.mode not in ["inference", "update", "summary", "report"]:
        logger.error("Invalid mode. Must be one of 'inference', 'update', or 'summary'.")
        return None
    return args

def main():
    args = parse_args()
    if not args:
        return

    init_logger()
    create_dirs()

    config = Config(CONFIG_PATH)
    model_trainer = ModelTrainer(config)

    if args.mode == "inference":
        if not args.file:
            logger.error("File path must be provided for inference mode.")
            return
        data_preparer = DataPreparer(config)
        data = pd.read_csv(args.file)
        X_transformed = data_preparer.transform_data(data)
        predictions = model_trainer.predict(X_transformed)
        data["predict"] = predictions
        output_path = os.path.join(PROCESSED_DATA_DIR, "inference_results.csv")
        data.to_csv(output_path, index=False)
        logger.info(f"Inference completed. Results saved to {output_path}")
        return output_path
    
    elif args.mode == "update":
        data_collector = DataCollector(config)
        if not data_collector.is_data_available():
            if not os.path.exists(os.path.join(RAW_DATA_DIR, "Sales.csv")):
                download_kaggle_dataset(DATASET_ID, RAW_DATA_DIR)
            data_collector.collect_initial_data(os.path.join(RAW_DATA_DIR, "Sales.csv"))
        data_preparer = DataPreparer(config)
        batch, batch_id = data_collector.get_next_batch()
        if batch_id < 0:
            return False
        X_train_prepared, X_test_prepared, y_train, y_test = data_preparer.prepare_data(
            batch, "Revenue", ["Date"]
        )
        train_result = None
        if batch_id == 0 or 1:
            model_trainer.train_models(X_train_prepared, y_train, X_test_prepared, y_test, batch_id)
        else:
            train_result = model_trainer.retrain_model(X_train_prepared, y_train, X_test_prepared, y_test, batch_id)
        logger.info(f"Model updated for batch {batch_id}")
        if train_result:
            print(json.dumps(train_result, indent=2))
        return True

    elif args.mode == "summary":
        summary: Dict = model_trainer.get_model_summary()
        summary_path = os.path.join(REPORTS_DIR, f"model_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Model summary saved to {summary_path}")
        return summary

    elif args.mode == "report":
        logger.info("Running report generation...")
        data_collector = DataCollector(config)
        data_preparer = DataPreparer(config)
        batch, batch_id = data_collector.get_latest_batch()
        if batch_id < 0:
            logger.error("No data available for report generation.")
            return
        X_train_prepared, X_test_prepared, y_train, y_test = data_preparer.prepare_data(
            batch, "Revenue", ["Date"]
        )
        model_visualizer = ModelVisualizer()
        model_visualizer.compare_models()
        model_visualizer.visualize_model(X_test_prepared, y_test)

        if not data_collector.generate_eda_report(batch, batch_id):
            return
    else:
        logger.error("Invalid mode. Please choose from: inference, update, or summary.")

if __name__ == "__main__":
    main()
