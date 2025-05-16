from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class PyTorchANN(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    
    def __init__(self, input_dim, epochs=100, batch_size=128, lr=0.001, patience=5):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._is_fitted = False  # Required for scikit-learn
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)
        
        # Initialize model
        self.model = PyTorchANN(self.input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss * 0.9999:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        
        self._is_fitted = True
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds
    
    def __sklearn_is_fitted__(self):
        """Return whether the estimator is fitted"""
        return hasattr(self, '_is_fitted') and self._is_fitted
    
    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'patience': self.patience
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self