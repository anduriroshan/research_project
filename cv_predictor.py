import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import root_mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from joblib import parallel_backend, Parallel, delayed
import os
from sklearn.model_selection import GridSearchCV
from pytorch_model import PyTorchRegressor, PyTorchWrapper
# Set number of cores to use
n_cores = max(1, os.cpu_count() - 1)
torch.set_num_threads(n_cores) # Set PyTorch to use multiple cores


def train_stacking_model(df, progress_callback=None):
    X = df[['Voltage', 'ScanRate']].values
    y = df['Current'].values

    # More robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Update progress if callback provided
    if progress_callback:
        progress_callback("Initializing machine learning models...", 0.1)

    # Define base models with optimized hyperparameters for speed
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Reduced from 200
        max_depth=8,       # Slightly reduced complexity
        min_samples_split=5,
        random_state=42,
        n_jobs=n_cores     # Utilize all cores
    )
    
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=200,  # Reduced from 500
        learning_rate=0.05, # Increased for faster convergence
        max_depth=5,
        num_leaves=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=n_cores      # Utilize all cores
    )
    
    pytorch_regressor = PyTorchRegressor(
        input_dim=X_train.shape[1],
        epochs=100,         # Reduced from 300
        batch_size=256,     # Larger batches for speed
        lr=0.005,           # Increased for faster convergence
        patience=5          # Reduced from 15
    )
    wrapped_pytorch = PyTorchWrapper(pytorch_regressor)
    # Update progress if callback provided
    if progress_callback:
        progress_callback("Building stacked ensemble model...", 0.2)

    # Use only the best final estimator to save time
    final_estimator = GradientBoostingRegressor(
        n_estimators=100,    # Reduced from 100
        learning_rate=0.05,  # Increased for faster convergence
        max_depth=3,
        random_state=42
    )
    
    # Update progress if callback provided
    if progress_callback:
        progress_callback("Training random forest model...", 0.3)
    
    model = StackingRegressor(
        estimators=[
            ('RF', rf_model),
            ('LGBM', lgbm_model),
            ('PyTorch', wrapped_pytorch),  # Keep PyTorch model for accuracy
        ],
        final_estimator=final_estimator,
        n_jobs=n_cores
    )
    
    # Use parallel backend for all sklearn operations
    with parallel_backend('threading', n_jobs=n_cores):
        # Update progress if callback provided
        if progress_callback:
            progress_callback("Fitting stacked ensemble model (this may take a moment)...", 0.4)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback("Evaluating model performance...", 0.8)
        
        # Final evaluation
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Store metrics
        metrics = {
            "rmse": rmse,
            "r2": r2
        }
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(f"Model training complete! RMSE: {rmse:.8f}, R²: {r2:.6f}", 1.0)

    return model, scaler, metrics

# --- Predict CV Data for a Scan Rate (Optimized for batch prediction) ---
def predict_cv_data(model, scaler, voltage_range, scan_rate):
    # Create input array more efficiently
    n_points = len(voltage_range)
    inputs = np.column_stack([
        voltage_range,
        np.full(n_points, scan_rate)
    ])
    
    X_scaled = scaler.transform(inputs)
    
    # Predict in batches for large voltage ranges
    batch_size = 10000
    if n_points > batch_size:
        predicted_current = np.zeros(n_points)
        for i in range(0, n_points, batch_size):
            end = min(i + batch_size, n_points)
            predicted_current[i:end] = model.predict(X_scaled[i:end])
    else:
        predicted_current = model.predict(X_scaled)
        
    return pd.DataFrame({
        'Voltage': voltage_range,
        'Predicted_Current': predicted_current
    })
