import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from data_processor import DataProcessor

class EquationSolver:
    def __init__(self, poly_degree=9):
        self.poly_degree = poly_degree
        self.processor = DataProcessor()

    def fit_polynomial(self, x, y):
        """Fits a polynomial of given degree to x and y data."""
        coeffs = np.polyfit(x, y, self.poly_degree)
        poly_func = np.poly1d(coeffs)
        return coeffs, poly_func(x)  # Return coefficients and fitted values

    def process_file(self, file_path):
        """Processes a file to extract anode and cathode data and fit polynomials."""
        _, anode_df, cathode_df = self.processor.get_cycle_data(file_path)

        # Fit polynomial to anode and cathode halves
        anode_coeffs, anode_fitted = self.fit_polynomial(anode_df["WE(1).Potential (V)"], anode_df["WE(1).Current (A)"])
        cathode_coeffs, cathode_fitted = self.fit_polynomial(cathode_df["WE(1).Potential (V)"], cathode_df["WE(1).Current (A)"])

        return {
            "anode": {"x": anode_df["WE(1).Potential (V)"], "y": anode_df["WE(1).Current (A)"], "fitted_y": anode_fitted, "coeffs": anode_coeffs},
            "cathode": {"x": cathode_df["WE(1).Potential (V)"], "y": cathode_df["WE(1).Current (A)"], "fitted_y": cathode_fitted, "coeffs": cathode_coeffs},
        }

    def extract_current_at_voltage(self, fitted_x, fitted_y, target_voltage=1):
        """Finds the current value at a specific voltage from the fitted curve."""
        closest_index = np.abs(fitted_x - target_voltage).argmin()
        return fitted_y[closest_index]

    def compute_k1_k2(self, scan_rates, current_values):
        """Solves for k1 and k2 using linear regression, converting scan rate from mV/s to V/s."""
        scan_rates_v = np.array(scan_rates) / 1000  # Convert from mV/s to V/s
        x = np.sqrt(scan_rates_v)  # v^(1/2)
        y = np.array(current_values) / x  # i/v^(1/2)

        k1, k2 = np.polyfit(x, y, 1)  # Fit y = k1 * x + k2
        return k1, k2, x, y

    def compute_contributions(self, scan_rates, k1, k2, currents):
        """Computes percentage contribution of capacitive and diffusion-controlled effects."""
        scan_rates_v = np.array(scan_rates) / 1000  # Convert from mV/s to V/s
        v_sqrt = np.sqrt(scan_rates_v)

        capacitive_effect = k1 * scan_rates_v  # k1 * v
        diffusion_effect = k2 * v_sqrt  # k2 * v^(1/2)

        total_current = np.array(currents)
        
        capacitive_percentage = (capacitive_effect / total_current) * 100
        diffusion_percentage = (diffusion_effect / total_current) * 100

        return capacitive_percentage, diffusion_percentage

    def process_all_scan_rates(self, file_paths, scan_rates, target_voltage=0.6):
        """Processes all scan rates, extracts current at target voltage, computes k1, k2, and contributions."""
        anode_currents = []
        cathode_currents = []

        for file_path in file_paths:
            results = self.process_file(file_path)

            # Extract current at target voltage for anode and cathode
            anode_current = self.extract_current_at_voltage(results["anode"]["x"].values, results["anode"]["fitted_y"], target_voltage)
            cathode_current = self.extract_current_at_voltage(results["cathode"]["x"].values, results["cathode"]["fitted_y"], target_voltage)

            anode_currents.append(anode_current)
            cathode_currents.append(cathode_current)

        # Compute k1, k2 for anode and cathode halves
        k1_anode, k2_anode, x_anode, y_anode = self.compute_k1_k2(scan_rates, anode_currents)
        k1_cathode, k2_cathode, x_cathode, y_cathode = self.compute_k1_k2(scan_rates, cathode_currents)

        # Compute percentage contributions
        anode_cap, anode_diff = self.compute_contributions(scan_rates, k1_anode, k2_anode, anode_currents)
        cathode_cap, cathode_diff = self.compute_contributions(scan_rates, k1_cathode, k2_cathode, cathode_currents)

        return {
        "anode": {
            "k1": k1_anode,
            "k2": k2_anode,
            "x": x_anode,  # Add this
            "y": y_anode,  # Add this
            "capacitive": anode_cap,
            "diffusion": anode_diff
        },
        "cathode": {
            "k1": k1_cathode,
            "k2": k2_cathode,
            "x": x_cathode,  # Add this
            "y": y_cathode,  # Add this
            "capacitive": cathode_cap,
            "diffusion": cathode_diff
        },
    }

    

    def plot_k1_k2(self, x, y, k1, k2, title):
        """Plots y = k1*x + k2 as a linear fit."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, color='blue', label="Actual Data")
        ax.plot(x, k1 * x + k2, color='red', linestyle="--", label=f"Fit: y = {k1:.4f}x + {k2:.4f}")
        ax.set_xlabel("v^(1/2)")
        ax.set_ylabel("i / v^(1/2)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)  # Display in Streamlit

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
