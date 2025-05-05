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
from sklearn.model_selection import GridSearchCV

# Improved PyTorch ANN Model
class PyTorchANN(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, epochs=200, batch_size=64, lr=0.001, patience=10):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.model = PyTorchANN(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scaler_x = RobustScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        # Scale features and target
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled).view(-1, 1)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
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
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
                best_weights = self.model.state_dict()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_weights)
                    break
        
        return self
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler_x.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = self.model(X_tensor).numpy().flatten()
        return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

def train_stacking_model(df):
    X = df[['Voltage', 'ScanRate']].values
    y = df['Current'].values

    # More robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Define base models with better hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    pytorch_regressor = PyTorchRegressor(
        input_dim=X_train.shape[1],
        epochs=300,
        batch_size=128,
        lr=0.0005,
        patience=15
    )

    # Try different final estimators
    final_estimators = [
        ('Ridge', RidgeCV()),
        ('Lasso', LassoCV(cv=5)),
        ('GBM', GradientBoostingRegressor(n_estimators=100))
    ]
    
    # Evaluate different stacking configurations
    best_score = float('inf')
    best_model = None
    
    for name, final_estimator in final_estimators:
        print(f"\nEvaluating {name} as final estimator...")
        
        # Use cross-validation for more reliable evaluation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = StackingRegressor(
                estimators=[
                    ('ANN', pytorch_regressor),
                    ('RF', rf_model),
                    ('LGBM', lgbm_model)
                ],
                final_estimator=final_estimator,
                n_jobs=-1
            )
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            score = root_mean_squared_error(y_val, preds)
            cv_scores.append(score)
        
        avg_score = np.mean(cv_scores)
        print(f"{name} CV RMSE: {avg_score:.4f}")
        
        if avg_score < best_score:
            best_score = avg_score
            best_model = model
    
    # Retrain best model on full training data
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    preds = best_model.predict(X_test)
    print("\nFinal Test Performance:")
    print("Test RMSE:", root_mean_squared_error(y_test, preds))
    print("Test R2:", r2_score(y_test, preds))

    return best_model, scaler

# --- Predict CV Data for a Scan Rate ---
def predict_cv_data(model, scaler, voltage_range, scan_rate):
    inputs = pd.DataFrame({
        'Voltage': voltage_range,
        'ScanRate': [scan_rate] * len(voltage_range)
    })
    X_scaled = scaler.transform(inputs)
    predicted_current = model.predict(X_scaled)
    return pd.DataFrame({
        'Voltage': voltage_range,
        'Predicted_Current': predicted_current.flatten()
    })