import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks,argrelextrema 
import numpy as np # For peak detection
import streamlit as st  # If you're using Streamlit

class DataProcessor:
    def __init__(self, storage_dir="stored_csvs"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.scan_rates = []
        self.file_map = {}

    def set_scan_rates(self, scan_rates):
        self.scan_rates = [float(rate) for rate in scan_rates]

    def get_scan_rates(self):
        return self.scan_rates

    def store_csv(self, file_path, scan_rate):
        stored_path = os.path.join(self.storage_dir, f"scan_{scan_rate}.csv")
        df = pd.read_excel(file_path)
        df.to_csv(stored_path, index=False)
        self.file_map[scan_rate] = stored_path
        return stored_path



    def extract_second_cycle(self, file_path):
        df = pd.read_csv(file_path)
        potential_col = 'WE(1).Potential (V)'
        current_col = 'WE(1).Current (A)'
        potential = df[potential_col].values

        # Find local minima
        minima_indices = argrelextrema(potential, np.less, order=20)[0]

        # Filter for valleys near 0V
        valley_indices = [i for i in minima_indices if potential[i] < 0.05]

        # Check index 0 manually
        if potential[0] < potential[1] and potential[0] < 0.05:
            valley_indices = [0] + valley_indices

        valley_indices = sorted(valley_indices)

        if len(valley_indices) < 3:
            raise ValueError(f"Not enough valleys to extract second full cycle. Found only {len(valley_indices)}.")

        start_idx = valley_indices[1]  # start of second cycle
        end_idx = valley_indices[2]    # end of second cycle

        second_cycle = df.iloc[start_idx:end_idx].reset_index(drop=True)
        return second_cycle[[potential_col, current_col]], valley_indices


    def split_anode_cathode(self, df):
        """Splits the provided second cycle data into anode and cathode halves by slope."""
        potential_col = 'WE(1).Potential (V)'
        current_col = 'WE(1).Current (A)'

        # Calculate the slope (potential difference) between points
        diff = df[potential_col].diff()

        # Use sign of slope to split data
        anode_df = df[diff > 0].reset_index(drop=True)     # Increasing potential
        cathode_df = df[diff < 0].reset_index(drop=True)   # Decreasing potential

        return anode_df, cathode_df


    def plot_all_full_curves(self, file_paths, scan_rates):
        """Plots full curves (both anode and cathode) for all scan rates and returns the figure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for file_path, scan_rate in zip(file_paths, scan_rates):
            try:
                cycle_df, _ = self.extract_second_cycle(file_path)
                ax.scatter(
                    cycle_df["WE(1).Potential (V)"],
                    cycle_df["WE(1).Current (A)"],
                    s=5, alpha=0.7, label=f"Scan Rate {scan_rate}"
                )
            except ValueError as e:
                print(f"Skipping {file_path}: {e}")  # Log the error
                st.warning(f"Skipping {os.path.basename(file_path)}: {e}")  # Inform the user in Streamlit
                continue  # Skip to the next file
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                st.error(f"File not found: {os.path.basename(file_path)}")
                continue

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title("Full Curve - All Scan Rates")
        ax.legend()
        ax.grid(True)
        return fig

    
    def plot_split_curves(self, file_path, title):
        """Plots anode and cathode curves separately and returns the figures."""

        try:
            cycle_df, _ = self.extract_second_cycle(file_path)
            anode_df, cathode_df = self.split_anode_cathode(cycle_df)

        except ValueError as e:
            raise ValueError(f"Data processing error for {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        fig_anode, ax_anode = plt.subplots(figsize=(8, 6))
        ax_anode.scatter(anode_df["WE(1).Potential (V)"], anode_df["WE(1).Current (A)"],
                        s=5, alpha=0.7, label="Anode Half")
        ax_anode.set_xlabel("Potential (V)")
        ax_anode.set_ylabel("Current (A)")
        ax_anode.set_title(f"{title} - Anode Half")
        ax_anode.legend()
        ax_anode.grid(True)

        fig_cathode, ax_cathode = plt.subplots(figsize=(8, 6))
        ax_cathode.scatter(cathode_df["WE(1).Potential (V)"], cathode_df["WE(1).Current (A)"],
                        s=5, alpha=0.7, label="Cathode Half")
        ax_cathode.set_xlabel("Potential (V)")
        ax_cathode.set_ylabel("Current (A)")
        ax_cathode.set_title(f"{title} - Cathode Half")
        ax_cathode.legend()
        ax_cathode.grid(True)

        return fig_anode, fig_cathode

    def plot_all_curves(self, file_paths, scan_rates):
        """Plots full, anode, and cathode curves separately for all scan rates and returns the figures."""

        fig_full, ax_full = plt.subplots(figsize=(8, 6))
        fig_anode, ax_anode = plt.subplots(figsize=(8, 6))
        fig_cathode, ax_cathode = plt.subplots(figsize=(8, 6))

        for file_path, scan_rate in zip(file_paths, scan_rates):
            try:
                cycle_df, _ = self.extract_second_cycle(file_path)
                anode_df, cathode_df = self.split_anode_cathode(cycle_df)


                # Full curve (both anode and cathode)
                ax_full.scatter(
                    cycle_df["WE(1).Potential (V)"],
                    cycle_df["WE(1).Current (A)"],
                    s=5, alpha=0.7, label=f"Scan Rate {scan_rate}"
                )

                # Anode curve (charging)
                ax_anode.scatter(
                    anode_df["WE(1).Potential (V)"],
                    anode_df["WE(1).Current (A)"],
                    s=5, alpha=0.7, label=f"Scan Rate {scan_rate}"
                )

                # Cathode curve (discharging)
                ax_cathode.scatter(
                    cathode_df["WE(1).Potential (V)"],
                    cathode_df["WE(1).Current (A)"],
                    s=5, alpha=0.7, label=f"Scan Rate {scan_rate}"
                )
            except ValueError as e:
                print(f"Skipping {file_path}: {e}")
                st.warning(f"Skipping {os.path.basename(file_path)}: {e}")
                continue
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                st.error(f"File not found: {os.path.basename(file_path)}")
                continue

        # Formatting plots
        for ax, title in zip(
            [ax_full, ax_anode, ax_cathode],
            ["Full Curve - All Scan Rates", "Anode Half (Charging) - All Scan Rates", "Cathode Half (Discharging) - All Scan Rates"]
        ):
            ax.set_xlabel("Potential (V)")
            ax.set_ylabel("Current (A)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

        return fig_full, fig_anode, fig_cathode

    def get_cycle_data(self, file_path):
        """Returns full cycle data, anode data, and cathode data."""
        try:
            cycle_df, _ = self.extract_second_cycle(file_path)  # ✅ Unpack here
            anode, cathode = self.split_anode_cathode(cycle_df)  # ✅ Now passing DataFrame
            return cycle_df, anode, cathode
        except ValueError as e:
            raise ValueError(f"Data processing error for {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
