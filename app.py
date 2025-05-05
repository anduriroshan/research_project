import streamlit as st
from data_processor import DataProcessor
from equation_solver import EquationSolver
from capacitance_analyzer import CapacitanceAnalyzer
from cv_predictor import train_stacking_model, predict_cv_data
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
if "cv_model" not in st.session_state:
    st.session_state.cv_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

processor = DataProcessor()
solver = EquationSolver(poly_degree=9)
capacitance_analyzer = CapacitanceAnalyzer(processor)

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

    # Sidebar: Clear Data Button
    if st.button("Clear All Data"):
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)
        st.session_state.clear()
        st.success("All uploaded files and scan rates have been cleared!")
        st.rerun()

# Main View Selector
selected_view = st.selectbox(
    "Select Visualization", 
    options=[
        "None",
        "View Graphs",
        "View Fitted Curves",
        "View k1 & k2 Graphs",
        "View EDLC & Pseudo-Capacitive Currents",
        "CV Prediction & Capacitance Analysis"  # Combined view
    ],
    key="view_selector"
)

# Display the selected view
if selected_view == "View Graphs":
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

elif selected_view == "View Fitted Curves":
    st.subheader("Fitted Curve Visualization")
    selected_scan_rate = st.selectbox("Select a Scan Rate for Fitted Curve", st.session_state.scan_rates, key="fitted_scan_rate")
    fitted_option = st.radio("Select Curve to Display:", ["Anode Half", "Cathode Half"], key="fitted_curve")

    file_path = st.session_state.file_map.get(selected_scan_rate)
    results = solver.process_file(file_path)

    if fitted_option == "Anode Half":
        solver.plot_fitted_curve(results["anode"]["x"], results["anode"]["y"], results["anode"]["fitted_y"], f"Anode Half - Fitted Curve (Scan {selected_scan_rate})")
    elif fitted_option == "Cathode Half":
        solver.plot_fitted_curve(results["cathode"]["x"], results["cathode"]["y"], results["cathode"]["fitted_y"], f"Cathode Half - Fitted Curve (Scan {selected_scan_rate})")

elif selected_view == "View k1 & k2 Graphs":
    st.subheader("k1 & k2 Computation")

    file_paths = list(st.session_state.file_map.values())
    scan_rates = st.session_state.scan_rates

    results = solver.process_all_scan_rates(file_paths, scan_rates, target_voltage=0.6)

    st.write(f"### Anode Half: k1 = {results['anode']['k1']:.4f}, k2 = {results['anode']['k2']:.4f}")
    solver.plot_k1_k2(
        results["anode"]["voltage"],
        results["anode"]["current"],
        results["anode"]["k1"],
        results["anode"]["k2"],
        "Anode Half - k1 & k2 Fit"
    )

    st.write(f"### Cathode Half: k1 = {results['cathode']['k1']:.4f}, k2 = {results['cathode']['k2']:.4f}")
    solver.plot_k1_k2(
        results["cathode"]["voltage"],
        results["cathode"]["current"],
        results["cathode"]["k1"],
        results["cathode"]["k2"],
        "Cathode Half - k1 & k2 Fit"
    )

    st.subheader("Capacitive vs Diffusion-Controlled Contributions")
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

elif selected_view == "View EDLC & Pseudo-Capacitive Currents":
    st.subheader("EDLC & Pseudo-Capacitive Current Visualization")

    file_paths = list(st.session_state.file_map.values())
    scan_rates = st.session_state.scan_rates

    results = solver.process_all_scan_rates(file_paths, scan_rates)

    for file_path, scan_rate in zip(file_paths, scan_rates):
        st.write(f"### Scan Rate: {scan_rate} mV/s")

        edlc_percentage_anode = results["anode"]["capacitive"][scan_rates.index(scan_rate)]
        pseudo_percentage_anode = results["anode"]["diffusion"][scan_rates.index(scan_rate)]
        edlc_percentage_cathode = results["cathode"]["capacitive"][scan_rates.index(scan_rate)]
        pseudo_percentage_cathode = results["cathode"]["diffusion"][scan_rates.index(scan_rate)]

        fig_anode, fig_cathode = solver.plot_edlc_pseudo_split_curves(
            file_path, f"Scan Rate {scan_rate}",
            edlc_percentage_anode,
            pseudo_percentage_anode,
            edlc_percentage_cathode,
            pseudo_percentage_cathode
        )

        st.pyplot(fig_anode)
        st.pyplot(fig_cathode)

elif selected_view == "CV Prediction & Capacitance Analysis":
    st.subheader("CV Curve Prediction and Capacitance Analysis")
    
    # Check if we have data to train the model
    if st.session_state.file_map and not st.session_state.cv_model:
        st.info("Training CV prediction model...")
        
        # Prepare training data from cathode halves
        all_data = []
        for scan_rate, file_path in st.session_state.file_map.items():
            try:
                full_df, anode_df, cathode_df = processor.get_cycle_data(file_path)
                df = pd.DataFrame({
                    'Voltage': cathode_df["WE(1).Potential (V)"],
                    'Current': cathode_df["WE(1).Current (A)"],
                    'ScanRate': scan_rate
                })
                all_data.append(df)
            except Exception as e:
                st.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        if all_data:
            training_df = pd.concat(all_data)
            
            # Train model
            model, scaler = train_stacking_model(training_df)
            st.session_state.cv_model = model
            st.session_state.scaler = scaler
            st.success("Model trained successfully on cathode half data!")
        else:
            st.error("Could not train model - no valid cathode data found")

    if st.session_state.cv_model:
        st.markdown("""
        Predict CV curves and calculate specific capacitance for new scan rates.
        The model uses your uploaded cathode half data to predict CV curves.
        """)
        
        # Input parameters
        col1, col2 = st.columns(2)
        with col1:
            scan_rate = st.number_input(
                "Scan Rate (mV/s)", 
                min_value=1, 
                max_value=1000, 
                value=50
            )
        with col2:
            mass = st.number_input(
                "Mass of active material (g)", 
                min_value=0.0001, 
                max_value=1.0, 
                value=0.01, 
                step=0.001,
                format="%.4f"
            )
        
        # Voltage range for prediction (using typical cathode range)
        voltage_range = st.slider(
            "Potential range (V)",
            min_value=-0.2,
            max_value=1.2,
            value=(-0.2, 0.6),  # Typical cathode range
            step=0.01
        )
        
        if st.button("Predict CV and Calculate Specific Capacitance"):
            with st.spinner("Predicting cathode curve and calculating capacitance..."):
                try:
                    # Use capacitance_analyzer to calculate based on predicted CV
                    result = capacitance_analyzer.analyze_predicted_cv(
                        scan_rate,
                        voltage_range,
                        st.session_state.cv_model,
                        st.session_state.scaler,
                        mass
                    )
                    
                    # Generate voltage points for prediction (for plotting)
                    voltage_points = np.linspace(voltage_range[0], voltage_range[1], 100)
                    
                    # Predict data for plotting
                    predicted_df = predict_cv_data(
                        st.session_state.cv_model,
                        st.session_state.scaler,
                        voltage_points,
                        scan_rate
                    )
                    
                    # Display results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Specific Capacitance", f"{result['specific_capacitance']:.2f} F/g")
                        st.metric("Potential Range", f"{result['potential_range']:.2f} V")
                    with col2:
                        st.metric("Capacitance", f"{result['capacitance']:.4f} F")
                        st.metric("Enclosed Area", f"{result['area']:.4f} A·V")
                    
                    # Plot predicted cathode curve
                    st.subheader("Predicted Cathode Curve")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(predicted_df['Voltage'], predicted_df['Predicted_Current'], 
                           'r-', linewidth=2, label='Predicted Cathode')
                    ax.set_xlabel('Potential (V)')
                    ax.set_ylabel('Current (A)')
                    ax.set_title(f'Predicted Cathode Curve at {scan_rate} mV/s')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please upload CV data files first to train the prediction model")