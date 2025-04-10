import json
import os
import pickle
import time
import pandas as pd
import numpy as np

from typing import Dict
from datetime import datetime
from matplotlib import pyplot as plt

from common.logger import logger
from common.config import Config
from common.config_static import BEST_MODEL_PATH, MODELS_DIR, MODELS_INFO_PATH, REPORTS_DIR

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class ModelTrainer:
    """Handles model training and hyperparameter tuning"""
    
    def __init__(self, config: Config):
        """Initialize model trainer"""
        self.config = config
        self.algorithms = self.config.get("model_training", "algorithms")
        self.hyperparameters = self.config.get("model_training", "hyperparameters")
        self.validation_method = self.config.get("model_training", "validation_method")
        self.cv_folds = self.config.get("model_training", "cv_folds")
        self.models_info = []
        self.models_path = MODELS_INFO_PATH
        self.best_model = None
        self.best_model_path = BEST_MODEL_PATH
        self.load_models_info()
    
    def load_models_info(self):
        """Load models info from file if exists"""
        if os.path.exists(self.models_path):
            with open(self.models_path, 'r') as file:
                self.models_info = json.load(file)
    
    def save_models_info(self):
        """Save models info to file"""
        with open(self.models_path, 'w') as file:
            json.dump(self.models_info, file, indent=2)
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_id: int = -1):
        """Train multiple models and find the best one"""
        models = []
        
        if "linear_regression" in self.algorithms:
            models.append(("linear_regression", LinearRegression()))
        
        if "knn" in self.algorithms:
            for n_neighbors in self.hyperparameters.get("knn", {}).get("n_neighbors", [5]):
                models.append((f"knn_n{n_neighbors}", KNeighborsRegressor(n_neighbors=n_neighbors)))
        
        if "decision_tree" in self.algorithms:
            for max_depth in self.hyperparameters.get("decision_tree", {}).get("max_depth", [None]):
                models.append((f"decision_tree_d{max_depth}", DecisionTreeRegressor(max_depth=max_depth)))
        
        results = []
        best_score = float('-inf')
        best_model_name = None
        
        for name, model in models:
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_scores = {}
            if self.validation_method == "cv":
                cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
            elif self.validation_method == "timeseries_cv":
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
            else:
                cv_rmse = rmse
            
            training_time = time.time() - start_time
            
            # Store model results
            model_info = {
                "name": name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "batch_id": batch_id,
                "metrics": {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "cv_rmse": float(cv_rmse)
                },
                "training_time": training_time,
                "model_path": os.path.join(MODELS_DIR, f"{name}_batch_{batch_id}.pkl")
            }
            
            # Save the model
            with open(model_info["model_path"], 'wb') as f:
                pickle.dump(model, f)
            
            results.append(model_info)
            
            # Update best model if this one is better
            if r2 > best_score:
                best_score = r2
                best_model_name = name
                self.best_model = model
            
            logger.info(f"Model {name} trained: R2 = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        # Save best model
        if self.best_model is not None:
            with open(self.best_model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info(f"Best model saved: {best_model_name} with R2 = {best_score:.4f}")
        
        # Update models info
        self.models_info.extend(results)
        self.save_models_info()
    
    def retrain_model(self, X_new: np.ndarray, y_new: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_id: int = -1):
        """Retrain existing model with new data"""
        if self.best_model is None:
            if os.path.exists(self.best_model_path):
                with open(self.best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
                logger.info(f"Loaded existing model for retraining: {type(self.best_model).__name__}")
            else:
                logger.error("No existing model to retrain")
                return {"error": "No existing model to retrain"}
        
        model_type = type(self.best_model).__name__
        
        start_time = time.time()
        
        if model_type == "LinearRegression":
            # For linear regression, we need to retrain from scratch
            # as scikit-learn doesn't support partial_fit for LinearRegression
            X_sample = X_new  # Using only new data for simplicity
            y_sample = y_new
            self.best_model.fit(X_sample, y_sample)
        
        elif model_type == "KNeighborsRegressor":
            # KNN doesn't support incremental learning, so we need to retrain
            X_sample = X_new
            y_sample = y_new
            self.best_model.fit(X_sample, y_sample)
        
        elif model_type == "DecisionTreeRegressor":
            # Decision trees don't support incremental learning either
            X_sample = X_new
            y_sample = y_new
            self.best_model.fit(X_sample, y_sample)
        
        else:
            logger.warning(f"Unknown model type: {model_type}, retraining from scratch")
            self.best_model.fit(X_new, y_new)
        
        # Get predictions
        y_pred = self.best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # Update model info
        model_info = {
            "name": f"retrained_{model_type}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "batch_id": batch_id,
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            },
            "training_time": training_time,
            "model_path": os.path.join(MODELS_DIR, f"retrained_{model_type}_batch_{batch_id}.pkl")
        }
        
        # Save the retrained model
        with open(self.best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Also save a version with batch ID
        with open(model_info["model_path"], 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Update models info
        self.models_info.append(model_info)
        self.save_models_info()
        
        logger.info(f"Model retrained: R2 = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        return {
            "batch_id": batch_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            },
            "training_time": training_time
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the best model"""
        if self.best_model is None:
            if os.path.exists(self.best_model_path):
                with open(self.best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
            else:
                raise ValueError("No trained model available for prediction")
        
        # Track prediction performance
        start_time = time.time()
        predictions = self.best_model.predict(X)
        prediction_time = time.time() - start_time
        
        # Log prediction performance
        logger.info(f"Predictions made for {(X.shape[0])} samples in {prediction_time:.4f} seconds")
        logger.info(f"Average prediction time per sample: {(prediction_time/X.shape[0])*1000:.2f} ms")
        
        return predictions

    def get_model_summary(self) -> Dict:
        """Generate a summary of model performance history"""
        if not self.models_info:
            return {"error": "No model history available"}
        
        # Extract metrics over time
        timestamps = []
        rmse_history = []
        r2_history = []
        mae_history = []
        model_types = []
        batch_ids = []
        
        for model_info in sorted(self.models_info, key=lambda x: x["batch_id"]):
            timestamps.append(model_info["timestamp"])
            rmse_history.append(model_info["metrics"]["rmse"])
            r2_history.append(model_info["metrics"]["r2"])
            mae_history.append(model_info["metrics"]["mae"])
            model_types.append(model_info["name"])
            batch_ids.append(model_info["batch_id"])
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            "Timestamp": timestamps,
            "Batch": batch_ids,
            "Model": model_types,
            "RMSE": rmse_history,
            "R2": r2_history,
            "MAE": mae_history
        })
        
        # Calculate improvement metrics
        if len(summary_df) > 1:
            initial_rmse = summary_df.iloc[0]["RMSE"]
            final_rmse = summary_df.iloc[-1]["RMSE"]
            rmse_improvement = ((initial_rmse - final_rmse) / initial_rmse) * 100
            
            initial_r2 = summary_df.iloc[0]["R2"]
            final_r2 = summary_df.iloc[-1]["R2"]
            r2_improvement = final_r2 - initial_r2  # R2 is already a percentage
        else:
            rmse_improvement = 0
            r2_improvement = 0
        
        # Get best model info
        best_model_idx = summary_df["R2"].idxmax()
        best_model = summary_df.iloc[best_model_idx]
        
        summary = {
            "model_count": len(summary_df),
            "batch_count": len(set(batch_ids)),
            "best_model": {
                "name": best_model["Model"],
                "batch": best_model["Batch"],
                "timestamp": best_model["Timestamp"],
                "metrics": {
                    "rmse": best_model["RMSE"],
                    "r2": best_model["R2"],
                    "mae": best_model["MAE"]
                }
            },
            "latest_model": {
                "name": summary_df.iloc[-1]["Model"],
                "batch": summary_df.iloc[-1]["Batch"],
                "timestamp": summary_df.iloc[-1]["Timestamp"],
                "metrics": {
                    "rmse": summary_df.iloc[-1]["RMSE"],
                    "r2": summary_df.iloc[-1]["R2"],
                    "mae": summary_df.iloc[-1]["MAE"]
                }
            },
            "improvement": {
                "rmse_reduction_percent": float(rmse_improvement),
                "r2_increase": float(r2_improvement)
            },
            "history": summary_df.to_dict(orient="records")
        }
        
        return summary
