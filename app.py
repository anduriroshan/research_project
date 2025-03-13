import streamlit as st
from data_processor import DataProcessor
import os
import shutil

st.title("Scan Rate Data Visualizer")
# Define the directory where files are stored
UPLOAD_DIR = "stored_csvs"

# Function to clear all uploaded files
def clear_uploaded_files():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)  # Delete the entire folder
        os.makedirs(UPLOAD_DIR)  # Recreate the empty folder
    st.session_state["scan_rates"] = []  # Clear scan rates in session
    st.session_state["file_map"] = {}  # Clear stored file map
    st.success("All uploaded files and scan rates have been cleared!")

# Set up storage directory for CSVs
STORAGE_DIR = "stored_csvs"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize session state
if "scan_rates" not in st.session_state:
    st.session_state.scan_rates = []
if "file_map" not in st.session_state:
    st.session_state.file_map = {}
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False

processor = DataProcessor(storage_dir=STORAGE_DIR)

# Step 1: Get scan rates
scan_rate_input = st.text_input("Enter scan rates (comma-separated):", key="scan_rate_input")
if scan_rate_input and not st.session_state.scan_rates:
    scan_rates = [float(rate.strip()) for rate in scan_rate_input.split(",") if rate.strip().isdigit()]
    st.session_state.scan_rates = scan_rates
    processor.set_scan_rates(scan_rates)
    st.success("Scan rates saved!")

# Step 2: Check stored files
stored_files = {rate: os.path.join(STORAGE_DIR, f"scan_{rate}.csv") for rate in st.session_state.scan_rates if os.path.exists(os.path.join(STORAGE_DIR, f"scan_{rate}.csv"))}

if stored_files:
    st.write("📁 Previously uploaded files found:")
    for rate, path in stored_files.items():
        st.write(f"- **Scan Rate {rate}**: {path}")

# Step 3: Upload files for missing scan rates
missing_scan_rates = [rate for rate in st.session_state.scan_rates if rate not in stored_files]
if missing_scan_rates and not st.session_state.upload_complete:
    for scan_rate in missing_scan_rates:
        if scan_rate not in st.session_state.file_map:
            file = st.file_uploader(f"Upload file for Scan Rate {scan_rate}:", type=["xlsx"], key=f"file_{scan_rate}")
            
            if file:
                file_path = os.path.join(STORAGE_DIR, f"scan_{scan_rate}.xlsx")
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                stored_csv_path = processor.store_csv(file_path, scan_rate)
                st.session_state.file_map[scan_rate] = stored_csv_path

    all_uploaded = all(rate in stored_files or rate in st.session_state.file_map for rate in st.session_state.scan_rates)
    if all_uploaded:
        st.session_state.upload_complete = True
        st.rerun()

# Step 4: Show confirmation message
if st.session_state.upload_complete:
    st.write("✅ All files uploaded and stored! Ready to analyze.")

    # Step 5: Select scan rate (Adding "All Scan Rates" option)
    scan_rate_options = ["All Scan Rates"] + list(st.session_state.scan_rates)
    selected_scan_rate = st.selectbox("Select a Scan Rate", scan_rate_options)

    plot_option = st.radio("Select Curve to Display:", ["Full Curve", "Anode Half", "Cathode Half"])

    file_paths = list(stored_files.values())  # All stored files
    scan_rates = st.session_state.scan_rates  # Corresponding scan rates

    if selected_scan_rate == "All Scan Rates":
        fig_full, fig_anode, fig_cathode = processor.plot_all_curves(file_paths, scan_rates)

        if plot_option == "Full Curve":
            st.pyplot(fig_full)
        elif plot_option == "Anode Half":
            st.pyplot(fig_anode)
        elif plot_option == "Cathode Half":
            st.pyplot(fig_cathode)
    else:
        file_path = stored_files.get(selected_scan_rate, st.session_state.file_map.get(selected_scan_rate))
        if plot_option == "Full Curve":
            fig = processor.plot_all_full_curves([file_path], [selected_scan_rate])
            st.pyplot(fig)
        elif plot_option == "Anode Half":
            fig_anode, _ = processor.plot_split_curves(file_path, f"Scan Rate {selected_scan_rate}")
            st.pyplot(fig_anode)
        elif plot_option == "Cathode Half":
            _, fig_cathode = processor.plot_split_curves(file_path, f"Scan Rate {selected_scan_rate}")
            st.pyplot(fig_cathode)


# Add the "Clear Data" button in the UI
if st.button("Clear All Data"):
    clear_uploaded_files()