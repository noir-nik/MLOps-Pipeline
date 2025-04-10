import json
import os
import pickle
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import tree
from common.logger import logger
from common.config_static import BEST_MODEL_PATH, MODELS_INFO_PATH, REPORTS_DIR

class ModelVisualizer:
    def __init__(self):
        self.best_model = None
        self.best_model_path = BEST_MODEL_PATH
        self.models_info = []
        self.models_path = MODELS_INFO_PATH
        self.load_models_info()

    def load_models_info(self):
        """Load models info from file if exists"""
        if os.path.exists(self.models_path):
            with open(self.models_path, 'r') as file:
                self.models_info = json.load(file)

    def get_feature_importance(self, feature_names=None):
        """Extract feature importance from the best model if possible"""
        if self.best_model is None:
            if os.path.exists(self.best_model_path):
                with open(self.best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
            else:
                logger.error("No existing model for feature importance analysis")
                return {"error": "No existing model available"}
        
        model_type = type(self.best_model).__name__
        feature_importance = {}
        
        try:
            if model_type == "LinearRegression":
                if feature_names is not None and len(feature_names) == len(self.best_model.coef_):
                    feature_importance = dict(zip(feature_names, abs(self.best_model.coef_)))
                else:
                    feature_importance = {f"feature_{i}": abs(coef) for i, coef in enumerate(self.best_model.coef_)}
            
            elif model_type == "DecisionTreeRegressor":
                if feature_names is not None and len(feature_names) == len(self.best_model.feature_importances_):
                    feature_importance = dict(zip(feature_names, self.best_model.feature_importances_))
                else:
                    feature_importance = {f"feature_{i}": imp for i, imp in enumerate(self.best_model.feature_importances_)}
            
            elif model_type == "KNeighborsRegressor":
                logger.info("KNN model doesn't provide direct feature importance")
                # Could implement a permutation importance approach here
                feature_importance = {"message": "KNN doesn't provide native feature importance"}
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            feature_importance = {"error": str(e)}
        
        return feature_importance

    def visualize_model(self, X_test=None, y_test=None, feature_names=None):
        """Generate visualizations for model interpretation"""
        if self.best_model is None:
            if os.path.exists(self.best_model_path):
                with open(self.best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
            else:
                logger.error("No existing model for visualization")
                return {"error": "No existing model available"}
        
        model_type = type(self.best_model).__name__
        visualization_path = os.path.join(REPORTS_DIR, "model_visualization")
        os.makedirs(visualization_path, exist_ok=True)
        
        try:
            # Feature importance visualization
            if model_type in ["LinearRegression", "DecisionTreeRegressor"]:
                feature_importance = self.get_feature_importance(feature_names)
                if not isinstance(feature_importance, dict) or "error" in feature_importance:
                    logger.warning("Could not generate feature importance visualization")
                else:
                    plt.figure(figsize=(10, 6))
                    # Sort by importance
                    sorted_features = dict(
                        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
                    )
                    plt.bar(sorted_features.keys(), sorted_features.values())
                    plt.xticks(rotation=90)
                    plt.title(f"Feature Importance for {model_type}")
                    plt.tight_layout()
                    feature_importance_path = os.path.join(visualization_path, f"{model_type}_feature_importance.png")
                    plt.savefig(feature_importance_path)
                    plt.close()
            
            # Decision Tree visualization
            if model_type == "DecisionTreeRegressor":
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(self.best_model.n_features_in_)]
                    
                plt.figure(figsize=(20, 10))
                tree.plot_tree(self.best_model, feature_names=feature_names, filled=True, rounded=True)
                plt.title("Decision Tree Structure")
                tree_viz_path = os.path.join(visualization_path, "decision_tree_structure.png")
                plt.savefig(tree_viz_path)
                plt.close()
            
            # KNN visualization if test data is provided
            if model_type == "KNeighborsRegressor" and X_test is not None and y_test is not None:
                # For KNN, we can visualize the nearest neighbors for a sample point
                if X_test.shape[1] > 2:
                    # Too many features for direct visualization, use PCA to reduce
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    X_reduced = pca.fit_transform(X_test)
                    
                    plt.figure(figsize=(10, 8))
                    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_test, cmap='viridis', alpha=0.7)
                    plt.colorbar(label='Target Value')
                    plt.title("KNN Data Distribution (PCA-reduced)")
                    plt.xlabel("PCA Component 1")
                    plt.ylabel("PCA Component 2")
                    knn_viz_path = os.path.join(visualization_path, "knn_data_distribution.png")
                    plt.savefig(knn_viz_path)
                    plt.close()
                else:
                    # Direct visualization possible
                    plt.figure(figsize=(10, 8))
                    plt.scatter(X_test[:, 0], X_test[:, 1] if X_test.shape[1] > 1 else y_test, 
                            c=y_test, cmap='viridis', alpha=0.7)
                    plt.colorbar(label='Target Value')
                    plt.title("KNN Data Distribution")
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2" if X_test.shape[1] > 1 else "Target")
                    knn_viz_path = os.path.join(visualization_path, "knn_data_distribution.png")
                    plt.savefig(knn_viz_path)
                    plt.close()
            
            # Regression performance visualization
            if X_test is not None and y_test is not None:
                y_pred = self.best_model.predict(X_test)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title(f"{model_type} Prediction Performance")
                perf_viz_path = os.path.join(visualization_path, f"{model_type}_performance.png")
                plt.savefig(perf_viz_path)
                plt.close()
                
                # Residuals plot
                plt.figure(figsize=(10, 6))
                residuals = y_test - y_pred
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel("Predicted Values")
                plt.ylabel("Residuals")
                plt.title("Residual Analysis")
                resid_viz_path = os.path.join(visualization_path, f"{model_type}_residuals.png")
                plt.savefig(resid_viz_path)
                plt.close()

            logger.info(f"Model visualization saved to {visualization_path}")
            return {
                "model_type": model_type,
                "visualization_dir": visualization_path,
                "status": "Success"
            }
            
        except Exception as e:
            logger.error(f"Error during model visualization: {str(e)}")
            return {"error": str(e)}

    def compare_models(self):
        """Compare performance metrics across all trained models"""
        if not self.models_info:
            logger.warning("No models found for comparison")
            return {"error": "No models available for comparison"}
        
        # Extract performance metrics for comparison
        model_names = []
        rmse_values = []
        r2_values = []
        mae_values = []
        training_times = []
        batch_ids = []
        
        for model_info in self.models_info:
            model_names.append(model_info["name"])
            rmse_values.append(model_info["metrics"]["rmse"])
            r2_values.append(model_info["metrics"]["r2"])
            mae_values.append(model_info["metrics"]["mae"])
            training_times.append(model_info["training_time"])
            batch_ids.append(model_info["batch_id"])
        
        comparison_dir = os.path.join(REPORTS_DIR, "model_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Performance metrics comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(model_names, r2_values)
        plt.xticks(rotation=90)
        plt.title("R2 Comparison")
        plt.tight_layout()
        
        plt.subplot(2, 2, 2)
        plt.bar(model_names, rmse_values)
        plt.xticks(rotation=90)
        plt.title("RMSE Comparison")
        plt.tight_layout()
        
        plt.subplot(2, 2, 3)
        plt.bar(model_names, mae_values)
        plt.xticks(rotation=90)
        plt.title("MAE Comparison")
        plt.tight_layout()
        
        plt.subplot(2, 2, 4)
        plt.bar(model_names, training_times)
        plt.xticks(rotation=90)
        plt.title("Training Time Comparison (seconds)")
        plt.tight_layout()
        
        comparison_path = os.path.join(comparison_dir, "model_metrics_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        
        # Create dataframe for CSV export
        comparison_df = pd.DataFrame({
            "Model": model_names,
            "Batch": batch_ids,
            "R2": r2_values,
            "RMSE": rmse_values,
            "MAE": mae_values,
            "Training_Time": training_times
        })
        
        csv_path = os.path.join(comparison_dir, "model_metrics_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        
        # Create performance for multiple batches
        if len(set(batch_ids)) > 1:
            try:
                # Sort by batch ID
                comparison_df = comparison_df.sort_values('Batch')
                
                # Group by model type (extracting base model name before parameters)
                comparison_df['Model_Type'] = comparison_df['Model'].apply(
                    lambda x: x.split('_')[0] if '_' in x else x
                )
                
                # Plot performance over time for each model type
                model_types = comparison_df['Model_Type'].unique()
                
                plt.figure(figsize=(12, 8))
                
                for model_type in model_types:
                    subset = comparison_df[comparison_df['Model_Type'] == model_type]
                    plt.plot(subset['Batch'], subset['R2'], marker='o', label=f"{model_type}")
                
                plt.xlabel("Batch ID")
                plt.ylabel("R2 Score")
                plt.title("Model Performance Evolution Over Batches")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                evolution_path = os.path.join(comparison_dir, "model_evolution.png")
                plt.savefig(evolution_path)
                plt.close()
                
            except Exception as e:
                logger.error(f"Error creating evolution plot: {str(e)}")
        
        logger.info(f"Model comparison saved to {comparison_dir}")
        return {
            "metrics_comparison": comparison_path,
            "csv_report": csv_path,
            "evolution_plot": evolution_path if len(set(batch_ids)) > 1 else None
        }
