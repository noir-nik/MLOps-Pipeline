import os
import yaml
from common.config_static import CONFIG_PATH
from common.logger import logger


class Config:
    """Configuration manager for the MLOps pipeline"""

    def __init__(self, config_path: str = CONFIG_PATH):
        """Initialize configuration from YAML file"""
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as file:
                    self.config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self.create_default_config()
                logger.info(f"Default configuration created at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.create_default_config()

    def create_default_config(self):
        """Create default configuration"""
        self.config = {
            "data_collection": {
                "batch_size": 1000,
                "data_quality_threshold": 0.8,
                "handle_duplicates": True,
            },
            "data_analysis": {
                "missing_threshold": 0.2,
                "correlation_threshold": 0.8,
                "detect_outliers": True,
            },
            "data_preparation": {
                "test_size": 0.2,
                "random_state": 42,
                "handle_categorical": True,
                "handle_numerical": True,
                "scaling": "standard",
            },
            "model_training": {
                "algorithms": ["linear_regression", "knn", "decision_tree"],
                "hyperparameters": {
                    "knn": {"n_neighbors": [3, 5, 9, 15]},
                    "decision_tree": {"max_depth": [3, 5, 7, 13]},
                },
                "validation_method": "timeseries_cv",
                "cv_folds": 5,
            },
            "model_validation": {
                "metrics": ["rmse", "mae", "r2"],
                "interpret_model": True,
            },
            "model_serving": {"monitor_performance": True, "version_control": True},
        }

        with open(self.config_path, "w") as file:
            yaml.dump(self.config, file)

    def get(self, section: str, key: str = None):
        """Get configuration value"""
        try:
            if key:
                return self.config[section][key]
            return self.config[section]
        except KeyError:
            logger.warning(f"Configuration key not found: {section}.{key}")
            return None
