from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone
from sklearn.utils.metaestimators import _BaseComposition

class PyTorchWrapper(_BaseComposition):
    """Wrapper to make PyTorchRegressor fully compatible with scikit-learn"""
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y):
        # Clone the estimator to ensure a fresh copy
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self
        
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def get_params(self, deep=True):
        return {'estimator': self.estimator}
    
    def set_params(self, **params):
        if 'estimator' in params:
            self.estimator = params['estimator']
        return self
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
    _estimator_type = "regressor"

    def __init__(self, input_dim, epochs=100, batch_size=128, lr=0.001, patience=5):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.criterion = nn.MSELoss()
        self.scaler_x = RobustScaler()
        self.scaler_y = StandardScaler() # Assuming StandardScaler is used for y as per your code

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)
        self.model = PyTorchANN(self.input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).view(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=False,
            num_workers=0
        )
        
        best_loss = float('inf')
        no_improve = 0
        best_weights = None # Initialize best_weights
        
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
            
            if epoch_loss < best_loss * 0.9999:
                best_loss = epoch_loss
                no_improve = 0
                best_weights = self.model.state_dict()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if best_weights is not None: # Ensure best_weights was set
                        self.model.load_state_dict(best_weights)
                    break
        
        # If loop finished without improvement for 'patience' epochs,
        # and best_weights was captured, load it.
        # This handles cases where the loop breaks early or completes all epochs.
        if best_weights is not None and (no_improve >= self.patience or epoch == self.epochs -1):
             self.model.load_state_dict(best_weights)

        return self
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler_x.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'patience': self.patience
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self