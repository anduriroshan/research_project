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

# Set number of cores to use
n_cores = max(1, os.cpu_count() - 1)
torch.set_num_threads(n_cores) # Set PyTorch to use multiple cores

# Improved PyTorch ANN Model with faster convergence
class PyTorchANN(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # Reduced network size
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),  # LeakyReLU converges faster than ReLU
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, epochs=100, batch_size=128, lr=0.001, patience=5):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Initialize later to avoid issues with joblib
        self.criterion = nn.MSELoss()
        self.scaler_x = RobustScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        # Initialize model here for better parallelization
        self.model = PyTorchANN(self.input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Scale features and target
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).view(-1, 1).to(self.device)
        
        # Create DataLoader with pin_memory for faster data transfer
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=False,  # Already moved tensors to device
            num_workers=0  # Using main process to avoid overhead with small datasets
        )
        
        # Training with early stopping
        best_loss = float('inf')
        no_improve = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(loader)
            
            # Early stopping
            if epoch_loss < best_loss * 0.9999:  # Small threshold to avoid numerical instability
                best_loss = epoch_loss
                no_improve = 0
                best_weights = self.model.state_dict()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.model.load_state_dict(best_weights)
                    break
        
        return self
        
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler_x.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

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
            ('PyTorch', pytorch_regressor),  # Keep PyTorch model for accuracy
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
