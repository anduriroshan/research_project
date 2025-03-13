import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.file_paths = []
        self.labels = []

    def load_files(self, file_paths, labels=None):
        """Store file paths and labels (if provided)."""
        self.file_paths = file_paths
        self.labels = labels if labels else [f"Scan Rate {i+1}" for i in range(len(file_paths))]

    def extract_second_cycle(self, file_path):
        """Extracts the second cycle from the Excel data."""
        df = pd.read_excel(file_path)
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

    def plot_scatter(self, file_path, label):
        """Returns a full scatter plot for a single file."""
        cycle_df = self.extract_second_cycle(file_path)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            cycle_df["WE(1).Potential (V)"],
            cycle_df["WE(1).Current (A)"],
            s=5, color="blue", alpha=0.7, label=label
        )

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title(f"Scatter Plot - {label}")
        ax.legend()
        ax.grid(True)

        return fig

    def plot_split_curves(self, file_path, label):
        """Returns two figures: one for anode (charging) and one for cathode (discharging) for a single file."""
        cycle_df = self.extract_second_cycle(file_path)
        anode_df, cathode_df = self.split_anode_cathode(cycle_df)

        fig_anode, ax_anode = plt.subplots(figsize=(8, 6))
        fig_cathode, ax_cathode = plt.subplots(figsize=(8, 6))

        # Plot anode curve
        ax_anode.scatter(
            anode_df["WE(1).Potential (V)"],
            anode_df["WE(1).Current (A)"],
            s=5, color="red", alpha=0.7, label=label
        )
        ax_anode.set_xlabel("Potential (V)")
        ax_anode.set_ylabel("Current (A)")
        ax_anode.set_title(f"Anode Half (Charging) - {label}")
        ax_anode.legend()
        ax_anode.grid(True)

        # Plot cathode curve
        ax_cathode.scatter(
            cathode_df["WE(1).Potential (V)"],
            cathode_df["WE(1).Current (A)"],
            s=5, color="green", alpha=0.7, label=label
        )
        ax_cathode.set_xlabel("Potential (V)")
        ax_cathode.set_ylabel("Current (A)")
        ax_cathode.set_title(f"Cathode Half (Discharging) - {label}")
        ax_cathode.legend()
        ax_cathode.grid(True)

        return fig_anode, fig_cathode
