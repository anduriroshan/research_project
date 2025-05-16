from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition # Can still be useful for some meta-estimator behaviors if needed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# PyTorchANN class remains the same
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
        return self.network

# PyTorchRegressor class remains the same (you can keep or remove the explicit _estimator_type here)
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor" # Can be kept for good measure

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
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)
        self.model = PyTorchANN(self.input_dim).to(self.device) # Initialize model in fit
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
        best_weights = self.model.state_dict() # Initialize with initial weights

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
                    self.model.load_state_dict(best_weights)
                    break
        # Ensure best weights are loaded if loop completes or breaks due to patience
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
        # Re-initialize model if relevant parameters change (e.g. input_dim),
        # or ensure this is handled if set_params is called post-init and pre-fit.
        # For StackingRegressor, clone usually handles re-initialization from scratch.
        return self


# MODIFIED PyTorchWrapper
class PyTorchWrapper(BaseEstimator, RegressorMixin): # Inherit BaseEstimator and RegressorMixin
    """
    Wrapper to make PyTorchRegressor fully compatible with scikit-learn,
    acting as a proper scikit-learn estimator itself.
    """
    _estimator_type = "regressor"
    def __init__(self, estimator=None, # Allow passing a pre-configured estimator
                       input_dim=2, epochs=100, batch_size=128, lr=0.001, patience=5): # Or params to create one
        
        # These are the parameters of the PyTorchWrapper itself
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        
        if estimator is not None:
            self.estimator = estimator
            # If estimator is provided, its params might override wrapper's params for the actual model
            # For simplicity, we'll assume if estimator is passed, it's fully configured.
            # Or, we can extract params from it if they are not set on wrapper:
            # self.input_dim = getattr(estimator, 'input_dim', input_dim)
            # ... and so on for other params
        else:
            # Estimator is not created here, but on fit, using the wrapper's parameters.
            # This allows GridSearchCV to set params on the wrapper, which are then used by PyTorchRegressor.
            self.estimator = None


    def fit(self, X, y):
        # Create and fit the internal PyTorchRegressor instance
        # Clone ensures that if self.estimator was pre-set, we work on a copy.
        # If self.estimator is None, we create a new PyTorchRegressor.
        if self.estimator is not None:
             # If an estimator instance was passed to __init__
            self.estimator_ = clone(self.estimator)
        else:
            # Create a new estimator using the wrapper's parameters
            self.estimator_ = PyTorchRegressor(
                input_dim=self.input_dim,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                patience=self.patience
            )
        
        self.estimator_.fit(X, y)
        self.is_fitted_ = True # For check_is_fitted on the wrapper
        return self

    def predict(self, X):
        check_is_fitted(self) # Checks self.is_fitted_
        return self.estimator_.predict(X)

    # get_params and set_params are now for the PyTorchWrapper's own parameters
    # These are automatically handled by BaseEstimator if defined in __init__
    
    # To be fully scikit-learn compliant and ensure check_is_fitted works robustly with this wrapper:
    def __sklearn_is_fitted__(self):
        # check_is_fitted will look for this method.
        # It should return True if the estimator has been fitted.
        return hasattr(self, 'is_fitted_') and self.is_fitted_