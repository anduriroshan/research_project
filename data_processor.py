import pandas as pd
import matplotlib.pyplot as plt
import os

class DataProcessor:
    def __init__(self, storage_dir="stored_csvs"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.scan_rates = []
        self.file_map = {}

    def set_scan_rates(self, scan_rates):
        """Store scan rates entered by the user."""
        self.scan_rates = [float(rate) for rate in scan_rates]

    def get_scan_rates(self):
        """Return the stored scan rates."""
        return self.scan_rates

    def store_csv(self, file_path, scan_rate):
        """Save a copy of the uploaded CSV file for future use."""
        stored_path = os.path.join(self.storage_dir, f"scan_{scan_rate}.csv")
        df = pd.read_excel(file_path)
        df.to_csv(stored_path, index=False)
        self.file_map[scan_rate] = stored_path
        return stored_path

    def extract_second_cycle(self, file_path):
        """Extracts the second cycle from the stored CSV data."""
        df = pd.read_csv(file_path)
        potential_col = 'WE(1).Potential (V)'
        current_col = 'WE(1).Current (A)'

        zero_indices = df[df[potential_col] < 0.001].index.tolist()
        if len(zero_indices) < 3:
            raise ValueError(f"Less than 3 cycles detected in {file_path}, check the data!")

        start_idx = zero_indices[1]
        end_idx = zero_indices[2]
        second_cycle = df.iloc[start_idx:end_idx].reset_index(drop=True)

        return second_cycle[[potential_col, current_col]]

    def split_anode_cathode(self, df):
        """Splits the second cycle into anode (charging) and cathode (discharging)."""
        potential_col = 'WE(1).Potential (V)'
        current_col = 'WE(1).Current (A)'

        peak_idx = df[potential_col].idxmax()
        anode_df = df.iloc[:peak_idx + 1]
        cathode_df = df.iloc[peak_idx + 1:]

        return anode_df, cathode_df

    def plot_all_full_curves(self, file_paths, scan_rates):
        """Plots full curves (both anode and cathode) for all scan rates."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for file_path, scan_rate in zip(file_paths, scan_rates):
            cycle_df = self.extract_second_cycle(file_path)
            ax.scatter(
                cycle_df["WE(1).Potential (V)"],
                cycle_df["WE(1).Current (A)"],
                s=5, alpha=0.7, label=f"Scan Rate {scan_rate}"
            )

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title("Full Curve - All Scan Rates")
        ax.legend()
        ax.grid(True)

        return fig
    def plot_split_curves(self, file_path, title):
     """Plots anode and cathode curves separately."""
     cycle_df = self.extract_second_cycle(file_path)
     anode_df, cathode_df = self.split_anode_cathode(cycle_df)

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
        """Plots full, anode, and cathode curves separately for all scan rates."""
        fig_full, ax_full = plt.subplots(figsize=(8, 6))
        fig_anode, ax_anode = plt.subplots(figsize=(8, 6))
        fig_cathode, ax_cathode = plt.subplots(figsize=(8, 6))

        for file_path, scan_rate in zip(file_paths, scan_rates):
            cycle_df = self.extract_second_cycle(file_path)
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
        full_cycle = self.extract_second_cycle(file_path)
        anode, cathode = self.split_anode_cathode(full_cycle)
        return full_cycle, anode, cathode
