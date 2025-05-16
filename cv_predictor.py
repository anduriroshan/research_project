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
from pytorch_model import PyTorchRegressor,PyTorchWrapper
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
    
    wrapped_pytorch = PyTorchWrapper(
        input_dim=X_train.shape[1], # Pass input_dim here
        epochs=100,
        batch_size=256,
        lr=0.005,
        patience=5
    )
# --- DEBUG PRINTS START ---
    if progress_callback: # Use existing progress_callback to indicate debugging step
        progress_callback("Debugging estimator types...", 0.15) # Arbitrary progress value

    print("--- Debugging Estimator Types (cv_predictor.py) ---")
    from sklearn.base import is_regressor, RegressorMixin # Ensure necessary imports
    import inspect

    # Define the models you are using in the StackingRegressor
    # Ensure these variable names (rf_model, lgbm_model, wrapped_pytorch)
    # match what you use in StackingRegressor's estimators list.

    # Example: Ensure rf_model, lgbm_model, wrapped_pytorch are defined as they will be used
    # rf_model = RandomForestRegressor(...)
    # lgbm_model = lgb.LGBMRegressor(...)
    # wrapped_pytorch = PyTorchWrapper(...) or PyTorchWrapper(PyTorchRegressor(...))

    estimators_to_check = {
        "RandomForest": rf_model,
        "LGBM": lgbm_model,
        "PyTorchWrapped": wrapped_pytorch  # This is the key one
    }

    for name, est_instance in estimators_to_check.items():
        print(f"\n[DEBUG] Checking: {name} (Instance Type: {type(est_instance).__name__})")

        # 1. Check _estimator_type attribute
        estimator_type_attr = getattr(est_instance, '_estimator_type', 'AttributeNotSet')
        print(f"  [DEBUG] {name}._estimator_type: {estimator_type_attr}")

        # 2. Check isinstance(est_instance, RegressorMixin)
        is_regressor_mixin_instance = isinstance(est_instance, RegressorMixin)
        print(f"  [DEBUG] isinstance({name}, RegressorMixin): {is_regressor_mixin_instance}")

        # 3. Check sklearn.base.is_regressor()
        try:
            is_sklearn_regressor_val = is_regressor(est_instance)
            print(f"  [DEBUG] sklearn.base.is_regressor({name}): {is_sklearn_regressor_val}")
        except Exception as e:
            print(f"  [DEBUG] Error calling sklearn.base.is_regressor({name}): {e}")

        # 4. Specifically for PyTorchWrapper, check its internal 'estimator' attribute if it exists
        # This depends on how your PyTorchWrapper is structured.
        # Case 1: PyTorchWrapper has an attribute like 'estimator' holding the PyTorchRegressor instance.
        if name == "PyTorchWrapped":
            # --- Start New Detailed PyTorchWrapped Checks ---
            print(f"  [DEBUG] Detailed check for PyTorchWrapped on cloud:")
            attr_value = getattr(est_instance, '_estimator_type', 'AttributeNotSet')
            expected_value = "regressor"
            print(f"    [DEBUG] Value of _estimator_type: '{attr_value}' (Type: {type(attr_value)})")
            print(f"    [DEBUG] Expected value: '{expected_value}' (Type: {type(expected_value)})")
            print(f"    [DEBUG] Direct comparison ('{attr_value}' == '{expected_value}'): {attr_value == expected_value}")

            # To ensure the is_regressor function is what we expect
            try:
                print(f"    [DEBUG] is_regressor function source: {inspect.getfile(is_regressor)}")
            except Exception as e_inspect:
                print(f"    [DEBUG] Could not get is_regressor source file: {e_inspect}")
            # --- End New Detailed PyTorchWrapped Checks ---

            if hasattr(est_instance, 'estimator'):
                internal_estimator = est_instance.estimator # The raw PyTorchRegressor object, if this structure is used
                if internal_estimator is not None:
                    print(f"  [DEBUG] Checking internal attribute 'estimator' of {name} (Instance Type: {type(internal_estimator).__name__})")
                    internal_estimator_type_attr = getattr(internal_estimator, '_estimator_type', 'AttributeNotSet')
                    print(f"    [DEBUG] internal_estimator._estimator_type: {internal_estimator_type_attr}")
                    is_internal_regressor_mixin = isinstance(internal_estimator, RegressorMixin)
                    print(f"    [DEBUG] isinstance(internal_estimator, RegressorMixin): {is_internal_regressor_mixin}")
                    try:
                        is_internal_sklearn_regressor = is_regressor(internal_estimator)
                        print(f"    [DEBUG] sklearn.base.is_regressor(internal_estimator): {is_internal_sklearn_regressor}")
                    except Exception as e:
                        print(f"    [DEBUG] Error calling sklearn.base.is_regressor(internal_estimator): {e}")

            # Case 2: PyTorchWrapper has a FITTED estimator attribute like 'estimator_' (with underscore)
            # This might not exist yet at the validation stage (before StackingRegressor.fit())
            if hasattr(est_instance, 'estimator_'):
                fitted_internal_estimator = est_instance.estimator_
                if fitted_internal_estimator is not None:
                    print(f"  [DEBUG] Checking internal FITTED attribute 'estimator_' of {name} (Instance Type: {type(fitted_internal_estimator).__name__})")
                    # ... (similar checks as above for fitted_internal_estimator)
                    fitted_estimator_type_attr = getattr(fitted_internal_estimator, '_estimator_type', 'AttributeNotSet')
                    print(f"    [DEBUG] fitted_internal_estimator._estimator_type: {fitted_estimator_type_attr}")
                    is_fitted_internal_regressor_mixin = isinstance(fitted_internal_estimator, RegressorMixin)
                    print(f"    [DEBUG] isinstance(fitted_internal_estimator, RegressorMixin): {is_fitted_internal_regressor_mixin}")
                    try:
                        is_fitted_internal_sklearn_regressor = is_regressor(fitted_internal_estimator)
                        print(f"    [DEBUG] sklearn.base.is_regressor(fitted_internal_estimator): {is_fitted_internal_sklearn_regressor}")
                    except Exception as e:
                         print(f"    [DEBUG] Error calling sklearn.base.is_regressor(fitted_internal_estimator): {e}")
                else:
                    print(f"  [DEBUG] Internal FITTED attribute 'estimator_' of {name} is None.")


    print("--- End Debugging Estimator Types ---")
# --- DEBUG PRINTS END ---

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
        verbose=0,
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
