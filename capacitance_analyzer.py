import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from cv_predictor import predict_cv_data 


class CapacitanceAnalyzer:
    def __init__(self, processor=None):
        """
        If processor is provided, it will be used to extract cycle data from real CV files.
        """
        self.processor = processor
        self.capacitance_data = None

    def calculate_enclosed_area(self, potential, current):
        """
        Calculate the enclosed area of a cyclic voltammetry curve using the Shoelace formula.
        """
        x = np.array(potential)
        y = np.array(current)
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return area

    def calculate_specific_capacitance(self, potential, current, scan_rate, mass=1.0):
        """
        Calculate specific capacitance from CV curve data.
        """
        potential_range = np.max(potential) - np.min(potential)
        area = self.calculate_enclosed_area(potential, current)
        scan_rate_v = scan_rate / 1000
        capacitance = area / (scan_rate_v * potential_range)
        specific_capacitance = capacitance / mass

        return {
            "scan_rate": scan_rate,
            "area": area,
            "potential_range": potential_range,
            "capacitance": capacitance,
            "specific_capacitance": specific_capacitance
        }

    def analyze_file(self, file_path, scan_rate, mass=1.0):
        """
        Analyze a single CV file using cathode half data.
        """
        try:
            full_df, anode_df, cathode_df = self.processor.get_cycle_data(file_path)
            result = self.calculate_specific_capacitance(
                cathode_df["WE(1).Potential (V)"],
                cathode_df["WE(1).Current (A)"],
                scan_rate,
                mass
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(full_df["WE(1).Potential (V)"], full_df["WE(1).Current (A)"],
                    'k-', alpha=0.5, label='Full cycle')
            ax.plot(cathode_df["WE(1).Potential (V)"], cathode_df["WE(1).Current (A)"],
                    'r-', linewidth=2, label='Cathode half')
            ax.set_xlabel('Potential (V)')
            ax.set_ylabel('Current (A)')
            ax.set_title(f'CV Curve at {scan_rate} mV/s')
            ax.grid(True)
            ax.legend()

            result['figure'] = fig
            return result
        except Exception as e:
            st.error(f"Error analyzing file {file_path}: {str(e)}")
            return None

    def analyze_all_files(self, file_paths, scan_rates, mass=1.0):
        """
        Analyze multiple CV files.
        """
        results = []
        for file_path, scan_rate in zip(file_paths, scan_rates):
            result = self.analyze_file(file_path, scan_rate, mass)
            if result:
                result.pop('figure', None)
                results.append(result)
        self.capacitance_data = pd.DataFrame(results)
        return self.capacitance_data

    def analyze_predicted_cv(self, scan_rate, voltage_range, model, scaler, mass=1.0):
        """
        Analyze predicted CV data (from model) and compute capacitance.
        """
        # Generate voltage points for prediction
        voltage_points = np.linspace(voltage_range[0], voltage_range[1], 100)
        
        # Predict using the correct function name
        predicted_cv = predict_cv_data(model, scaler, voltage_points, scan_rate)
        
        voltage_pred = predicted_cv["Voltage"]
        current_pred = predicted_cv["Predicted_Current"]  # Changed to match the returned column name
        
        result = self.calculate_specific_capacitance(voltage_pred, current_pred, scan_rate, mass)
        return result