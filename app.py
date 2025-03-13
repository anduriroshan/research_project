import streamlit as st
from data_processor import DataProcessor
import os

st.title("Scan Rate Data Visualizer")

# File uploader
uploaded_files = st.file_uploader("Upload up to 5 Excel files", type=["xlsx"], accept_multiple_files=True)

if "scan_data" not in st.session_state:
    st.session_state.scan_data = {}

if uploaded_files:
    processor = DataProcessor()
    file_paths = []

    # Save files temporarily
    for i, file in enumerate(uploaded_files):
        file_path = f"temp_file_{i}.xlsx"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)

    # Load files
    processor.load_files(file_paths)
    for i, file_path in enumerate(file_paths):
        label = f"Scan Rate {i+1}"
        st.session_state.scan_data[label] = file_path

    st.success("Files uploaded! Select a scan rate to view.")

# Select scan rate
if st.session_state.scan_data:
    scan_rate_selected = st.selectbox("Select Scan Rate", list(st.session_state.scan_data.keys()))

    if scan_rate_selected:
        file_path = st.session_state.scan_data[scan_rate_selected]

        # Select graph type
        plot_option = st.radio("Select Curve to Display:", ["Full Curve", "Anode Half", "Cathode Half"])

        processor = DataProcessor()  # Create processor instance

        if plot_option == "Full Curve":
            fig = processor.plot_scatter(file_path, scan_rate_selected)  # ✅ Pass file_path & label
            st.pyplot(fig)
        elif plot_option == "Anode Half":
            fig_anode, _ = processor.plot_split_curves(file_path, scan_rate_selected)  # ✅ Pass file_path & label
            st.pyplot(fig_anode)
        elif plot_option == "Cathode Half":
            _, fig_cathode = processor.plot_split_curves(file_path, scan_rate_selected)  # ✅ Pass file_path & label
            st.pyplot(fig_cathode)

# Clean up temporary files
if uploaded_files:
    for file_path in file_paths:
        os.remove(file_path)
