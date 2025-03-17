import streamlit as st
from data_processor import DataProcessor
from equation_solver import EquationSolver
import os
import shutil

st.title("Scan Rate Data Visualizer")

# Define the directory where files are stored
UPLOAD_DIR = "stored_csvs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state variables
if "scan_rates" not in st.session_state:
    st.session_state.scan_rates = []
if "file_map" not in st.session_state:
    st.session_state.file_map = {}
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False
if "show_graphs" not in st.session_state:
    st.session_state.show_graphs = False  # For actual graphs
if "show_fitted_graphs" not in st.session_state:
    st.session_state.show_fitted_graphs = False  # For fitted curves

processor = DataProcessor()
solver = EquationSolver(poly_degree=9)  # Polynomial degree 9

# Sidebar: Scan Rate Input
# Sidebar: Scan Rate Input
with st.sidebar:
    scan_rate_input = st.text_input("Enter Scan Rates (comma-separated):")
    if scan_rate_input and not st.session_state.scan_rates:
        scan_rates = [float(rate.strip()) for rate in scan_rate_input.split(",") if rate.strip().isdigit()]
        st.session_state.scan_rates = scan_rates
        processor.set_scan_rates(scan_rates)
        st.success("Scan rates saved!")

    # Sidebar: File Uploaders (only if files are missing)
    missing_scan_rates = [rate for rate in st.session_state.scan_rates if rate not in st.session_state.file_map]

    if missing_scan_rates:
        for scan_rate in missing_scan_rates:
            file = st.file_uploader(f"Upload file for Scan Rate {scan_rate}:", type=["xlsx"], key=f"file_{scan_rate}")
            if file:
                file_path = os.path.join(UPLOAD_DIR, f"scan_{scan_rate}.xlsx")
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                stored_csv_path = processor.store_csv(file_path, scan_rate)
                st.session_state.file_map[scan_rate] = stored_csv_path

        if all(rate in st.session_state.file_map for rate in st.session_state.scan_rates):
            st.session_state.upload_complete = True
            st.rerun()

    # Sidebar: Clear Data Button (Ensure it doesn't run on page load)
    if st.button("Clear All Data"):
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)
        st.session_state.clear()
        st.success("All uploaded files and scan rates have been cleared!")
        st.rerun()


# Show Graphs Button
if not st.session_state.show_graphs:
    if st.button("View Graphs"):
        st.session_state.show_graphs = True
        st.rerun()
else:
    if st.button("Close Graphs"):
        st.session_state.show_graphs = False
        st.rerun()

# Show Fitted Curves Button
if not st.session_state.show_fitted_graphs:
    if st.button("View Fitted Curves"):
        st.session_state.show_fitted_graphs = True
        st.rerun()
else:
    if st.button("Close Fitted Curves"):
        st.session_state.show_fitted_graphs = False
        st.rerun()


# Show k1 & k2 Graphs Button
if "show_k1_k2_graphs" not in st.session_state:
    st.session_state.show_k1_k2_graphs = False

if not st.session_state.show_k1_k2_graphs:
    if st.button("View k1 & k2 Graphs"):
        st.session_state.show_k1_k2_graphs = True
        st.rerun()
else:
    if st.button("Close k1 & k2 Graphs"):
        st.session_state.show_k1_k2_graphs = False
        st.rerun()

# k1 & k2 Graph Visualization
if st.session_state.show_k1_k2_graphs:
    st.subheader("k1 & k2 Computation")

    file_paths = list(st.session_state.file_map.values())
    scan_rates = st.session_state.scan_rates

    results = solver.process_all_scan_rates(file_paths, scan_rates, target_voltage=0.6)

    st.write(f"### Anode Half: k1 = {results['anode']['k1']:.4f}, k2 = {results['anode']['k2']:.4f}")
    solver.plot_k1_k2(results["anode"]["x"], results["anode"]["y"], results["anode"]["k1"], results["anode"]["k2"], "Anode Half - k1 & k2 Fit")

    st.write(f"### Cathode Half: k1 = {results['cathode']['k1']:.4f}, k2 = {results['cathode']['k2']:.4f}")
    solver.plot_k1_k2(results["cathode"]["x"], results["cathode"]["y"], results["cathode"]["k1"], results["cathode"]["k2"], "Cathode Half - k1 & k2 Fit")

if st.session_state.show_k1_k2_graphs:
    st.subheader("Capacitive vs Diffusion-Controlled Contributions")

    file_paths = list(st.session_state.file_map.values())
    scan_rates = st.session_state.scan_rates

    results = solver.process_all_scan_rates(file_paths, scan_rates, target_voltage=0.6)

    for i, scan_rate in enumerate(scan_rates):
        st.write(f"### Scan Rate: {scan_rate} mV/s")
        
        # Anode Half
        anode_cap = results["anode"]["capacitive"][i]
        anode_diff = results["anode"]["diffusion"][i]
        st.write(f"**Anode:** Capacitive: {anode_cap:.2f}%, Diffusion-Controlled: {anode_diff:.2f}%")

        # Cathode Half
        cathode_cap = results["cathode"]["capacitive"][i]
        cathode_diff = results["cathode"]["diffusion"][i]
        st.write(f"**Cathode:** Capacitive: {cathode_cap:.2f}%, Diffusion-Controlled: {cathode_diff:.2f}%")


# Graph Visualization Section
if st.session_state.show_graphs:
    st.subheader("Graph Visualization")
    scan_rate_options = ["All Scan Rates"] + list(st.session_state.scan_rates)
    selected_scan_rate = st.selectbox("Select a Scan Rate", scan_rate_options, key="graph_scan_rate")
    plot_option = st.radio("Select Curve to Display:", ["Full Curve", "Anode Half", "Cathode Half"], key="graph_curve")

    file_paths = list(st.session_state.file_map.values())
    scan_rates = st.session_state.scan_rates

    if selected_scan_rate == "All Scan Rates":
        fig_full, fig_anode, fig_cathode = processor.plot_all_curves(file_paths, scan_rates)
        if plot_option == "Full Curve":
            st.pyplot(fig_full)
        elif plot_option == "Anode Half":
            st.pyplot(fig_anode)
        elif plot_option == "Cathode Half":
            st.pyplot(fig_cathode)
    else:
        file_path = st.session_state.file_map.get(selected_scan_rate)
        if plot_option == "Full Curve":
            fig = processor.plot_all_full_curves([file_path], [selected_scan_rate])
            st.pyplot(fig)
        elif plot_option == "Anode Half":
            fig_anode, _ = processor.plot_split_curves(file_path, f"Scan Rate {selected_scan_rate}")
            st.pyplot(fig_anode)
        elif plot_option == "Cathode Half":
            _, fig_cathode = processor.plot_split_curves(file_path, f"Scan Rate {selected_scan_rate}")
            st.pyplot(fig_cathode)

if st.session_state.show_fitted_graphs:
    st.subheader("Fitted Curve Visualization")
    selected_scan_rate = st.selectbox("Select a Scan Rate for Fitted Curve", st.session_state.scan_rates, key="fitted_scan_rate")
    fitted_option = st.radio("Select Curve to Display:", ["Anode Half", "Cathode Half"], key="fitted_curve")

    file_path = st.session_state.file_map.get(selected_scan_rate)
    results = solver.process_file(file_path)

    if fitted_option == "Anode Half":
        solver.plot_fitted_curve(results["anode"]["x"], results["anode"]["y"], results["anode"]["fitted_y"], f"Anode Half - Fitted Curve (Scan {selected_scan_rate})")
    elif fitted_option == "Cathode Half":
        solver.plot_fitted_curve(results["cathode"]["x"], results["cathode"]["y"], results["cathode"]["fitted_y"], f"Cathode Half - Fitted Curve (Scan {selected_scan_rate})")
