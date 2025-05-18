import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from data_processor import DataProcessor
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
        """Solves for k1 and k2 using sklearn's LinearRegression and returns R² score."""
        scan_rates_v = np.array(scan_rates) / 1000  # Convert from mV/s to V/s
        x = np.sqrt(scan_rates_v).reshape(-1, 1)     # v^(1/2), reshaped for sklearn
        y = (np.array(current_values) / np.sqrt(scan_rates_v)).reshape(-1, 1)  # i / v^(1/2)

        model = LinearRegression()
        model.fit(x, y)

        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)

        k1 = float(model.coef_)
        k2 = float(model.intercept_)

        return k1, k2, x.flatten(), y.flatten(), r2

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
        k1_anode, k2_anode, x_anode, y_anode, r2_anode = self.compute_k1_k2(scan_rates, anode_currents)
        k1_cathode, k2_cathode, x_cathode, y_cathode, r2_cathode = self.compute_k1_k2(scan_rates, cathode_currents)


        # Compute percentage contributions
        anode_cap, anode_diff = self.compute_contributions(scan_rates, k1_anode, k2_anode, anode_currents)
        cathode_cap, cathode_diff = self.compute_contributions(scan_rates, k1_cathode, k2_cathode, cathode_currents)

        return {
    "anode": {
        "k1": k1_anode,
        "k2": k2_anode,
        "voltage": x_anode,
        "current": y_anode,
        "capacitive": anode_cap,
        "diffusion": anode_diff,
        "r2": r2_anode
    },
    "cathode": {
        "k1": k1_cathode,
        "k2": k2_cathode,
        "voltage": x_cathode,
        "current": y_cathode,
        "capacitive": cathode_cap,
        "diffusion": cathode_diff,
        "r2": r2_cathode
    },
}


    def plot_edlc_pseudo_split_curves(self, file_path, title, edlc_percentage_anode, pseudo_percentage_anode, edlc_percentage_cathode, pseudo_percentage_cathode):
        """Plots EDLC and pseudo-capacitive currents separately for anode and cathode."""
        cycle_df, _ = self.processor.extract_second_cycle(file_path)
        anode_df, cathode_df = self.processor.split_anode_cathode(cycle_df)

        # Ensure edlc_percentage_anode is a single scalar value
        if not np.isscalar(edlc_percentage_anode):
            edlc_percentage_anode = edlc_percentage_anode[0]  # Take the first value

        if not np.isscalar(pseudo_percentage_anode):
            pseudo_percentage_anode = pseudo_percentage_anode[0]

        if not np.isscalar(edlc_percentage_cathode):
            edlc_percentage_cathode = edlc_percentage_cathode[0]

        if not np.isscalar(pseudo_percentage_cathode):
            pseudo_percentage_cathode = pseudo_percentage_cathode[0]

        # Now, the multiplication will work correctly
        anode_df["EDLC_Current"] = (edlc_percentage_anode / 100) * anode_df["WE(1).Current (A)"]
        anode_df["Pseudo_Current"] = (pseudo_percentage_anode / 100) * anode_df["WE(1).Current (A)"]

        cathode_df["EDLC_Current"] = (edlc_percentage_cathode / 100) * cathode_df["WE(1).Current (A)"]
        cathode_df["Pseudo_Current"] = (pseudo_percentage_cathode / 100) * cathode_df["WE(1).Current (A)"]

        # Plot Anode Half
        fig_anode, ax_anode = plt.subplots(figsize=(8, 6))
        ax_anode.scatter(anode_df["WE(1).Potential (V)"], anode_df["WE(1).Current (A)"], s=5, alpha=0.7, label="Total Anode Current", color="black")
        ax_anode.scatter(anode_df["WE(1).Potential (V)"], anode_df["EDLC_Current"], s=5, alpha=0.7, label="EDLC Current", color="red")
        ax_anode.scatter(anode_df["WE(1).Potential (V)"], anode_df["Pseudo_Current"], s=5, alpha=0.7, label="Pseudo Current", color="blue")
        ax_anode.set_xlabel("Potential (V)")
        ax_anode.set_ylabel("Current (A)")
        ax_anode.set_title(f"{title} - Anode Half")
        ax_anode.legend()
        ax_anode.grid(True)

        # Plot Cathode Half
        fig_cathode, ax_cathode = plt.subplots(figsize=(8, 6))
        ax_cathode.scatter(cathode_df["WE(1).Potential (V)"], cathode_df["WE(1).Current (A)"], s=5, alpha=0.7, label="Total Cathode Current", color="black")
        ax_cathode.scatter(cathode_df["WE(1).Potential (V)"], cathode_df["EDLC_Current"], s=5, alpha=0.7, label="EDLC Current", color="red")
        ax_cathode.scatter(cathode_df["WE(1).Potential (V)"], cathode_df["Pseudo_Current"], s=5, alpha=0.7, label="Pseudo Current", color="blue")
        ax_cathode.set_xlabel("Potential (V)")
        ax_cathode.set_ylabel("Current (A)")
        ax_cathode.set_title(f"{title} - Cathode Half")
        ax_cathode.legend()
        ax_cathode.grid(True)

        return fig_anode, fig_cathode

    def plot_combined_edlc_pseudo_curve(self, file_path, edlc_percentage_anode, pseudo_percentage_anode,
                                        edlc_percentage_cathode, pseudo_percentage_cathode):
        import matplotlib.pyplot as plt
        import numpy as np

        # Use full cycle directly
        cycle_df, _ = self.processor.extract_second_cycle(file_path)

        voltage = cycle_df["WE(1).Potential (V)"].values
        current = cycle_df["WE(1).Current (A)"].values

        # Calculate average contribution percentages
        edlc_frac = (edlc_percentage_anode + edlc_percentage_cathode) / 200
        pseudo_frac = (pseudo_percentage_anode + pseudo_percentage_cathode) / 200

        # Compute component currents
        edlc_current = edlc_frac * current
        pseudo_current = pseudo_frac * current

        # Compute EDLC % of total current (optional info)
        area_total = np.trapz(np.abs(current), voltage)
        area_edlc = np.trapz(np.abs(edlc_current), voltage)
        area_pseudo = np.trapz(np.abs(pseudo_current), voltage)
        percent_edlc = 100 * area_edlc / area_total
        percent_pseudo = 100 * area_pseudo / area_total

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot total current from full cycle (normal + reverse scan)
        

        # Fill areas for EDLC and Pseudo currents
        #ax.fill_between(voltage, pseudo_current,color='yellow', label='Pseudocapacitive')

        ax.fill_between(voltage, edlc_current,color='green', label='EDLC')
        ax.plot(voltage, current, color='black', linewidth=1.5, label='Total Current')

        # Add contribution text
        ax.text(min(voltage) + 0.05 * (max(voltage) - min(voltage)),
                np.max(np.abs(current)) * 0.8,
                f"EDLC: {percent_edlc:.2f}%\nPseudo: {100-percent_edlc:.2f}%",
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7))

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title("Combined EDLC + Pseudocapacitive Contribution")
        ax.set_ylim(-1.1 * np.max(np.abs(current)), 1.1 * np.max(np.abs(current)))
        ax.set_xlim(min(voltage), max(voltage))
        ax.legend(['Legend'], loc=4)
        ax.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

    def plot_k1_k2(self, x, y, k1, k2,r2, title):
        """Plots y = k1*x + k2 as a linear fit."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, color='blue', label="Actual Data")
        ax.plot(x, k1 * x + k2, color='red', linestyle="--", label=f"Fit: y = {k1:.4f}x + {k2:.4f}\nR² = {r2:.4f}")
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
