import numpy as np
import matplotlib.pyplot as plt
from data_processor import DataProcessor
import streamlit as st

class EquationSolver:
    def __init__(self, poly_degree=25):
        self.poly_degree = poly_degree
        self.processor = DataProcessor()

    def fit_polynomial(self, x, y):
        """Fits a polynomial of given degree to x and y data."""
        coeffs = np.polyfit(x, y, self.poly_degree)
        poly_func = np.poly1d(coeffs)
        y_fitted = poly_func(x)
        return coeffs, y_fitted

    def process_file(self, file_path):
        """Processes a file to extract anode and cathode data and fit polynomials."""
        full_cycle, anode_df, cathode_df = self.processor.get_cycle_data(file_path)

        # Fit polynomial to anode (charging) curve
        anode_x = anode_df["WE(1).Potential (V)"].values
        anode_y = anode_df["WE(1).Current (A)"].values
        anode_coeffs, anode_fitted = self.fit_polynomial(anode_x, anode_y)

        # Fit polynomial to cathode (discharging) curve
        cathode_x = cathode_df["WE(1).Potential (V)"].values
        cathode_y = cathode_df["WE(1).Current (A)"].values
        cathode_coeffs, cathode_fitted = self.fit_polynomial(cathode_x, cathode_y)

        return {
            "anode": {"x": anode_x, "y": anode_y, "fitted_y": anode_fitted, "coeffs": anode_coeffs},
            "cathode": {"x": cathode_x, "y": cathode_y, "fitted_y": cathode_fitted, "coeffs": cathode_coeffs},
        }

    def plot_fitted_curve(self, x, y_actual, y_fitted, title):
        """Plots actual vs fitted curve and displays in Streamlit."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y_actual, s=12, alpha=0.6, label="Actual Data", color='red')
        ax.plot(x, y_fitted, color='yellow', label="Fitted Curve")
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig) 
