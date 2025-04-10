import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple
from common.logger import logger
from common.config import Config
from common.config_static import MODELS_DIR

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class DataPreparer:
    """Prepares data for model training"""
    
    def __init__(self, config: Config):
        """Initialize data preparer"""
        self.config = config
        self.test_size = self.config.get("data_preparation", "test_size")
        self.random_state = self.config.get("data_preparation", "random_state")
        self.handle_categorical = self.config.get("data_preparation", "handle_categorical")
        self.handle_numerical = self.config.get("data_preparation", "handle_numerical")
        self.scaling = self.config.get("data_preparation", "scaling")
        self.preprocessor = None
        self.preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, drop_columns: list = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""

        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Drop columns (if specified)
        X = X.drop(columns=drop_columns)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Build preprocessing pipeline if not already built
        if self.preprocessor is None:
            self.build_preprocessor(X_train)
        
        # Apply preprocessing
        X_train_prepared = self.preprocessor.transform(X_train)
        X_test_prepared = self.preprocessor.transform(X_test)
        
        logger.info(f"Data prepared: X_train shape = {X_train_prepared.shape}, X_test shape = {X_test_prepared.shape}")
        return X_train_prepared, X_test_prepared, y_train, y_test
    
    def build_preprocessor(self, X: pd.DataFrame):
        """Build data preprocessing pipeline"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        transformers = []
        
        # Categorical features preprocessing
        if self.handle_categorical and categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('categorical', categorical_transformer, categorical_cols))
        
        # Numerical features preprocessing
        if self.handle_numerical and numerical_cols:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler() if self.scaling else 'passthrough')
            ])
            transformers.append(('numerical', numerical_transformer, numerical_cols))
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(X)
        
        # Save preprocessor
        with open(self.preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        logger.info("Preprocessor built and saved")
    
    def load_preprocessor(self):
        """Load preprocessor from file"""
        if os.path.exists(self.preprocessor_path):
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded")
            return True
        return False
    
    def transform_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using the built preprocessor"""
        if self.preprocessor is None:
            if not self.load_preprocessor():
                raise ValueError("Preprocessor not available. Train model first.")
        
        return self.preprocessor.transform(df)
