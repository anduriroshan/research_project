import tensorflow as tf
import keras
from keras import layers
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["RootMeanSquaredError"])
    return model

def get_keras_regressor(input_dim, epochs=100, batch_size=128):
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    return KerasRegressor(
        model=build_model,
        model__input_dim=input_dim,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

