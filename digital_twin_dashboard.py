import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
import numpy as np
import os
import glob
from typing import Dict, Any, List

# --- CONFIGURATION & CONSTANTS ---
WINDOW_SIZE = 300
CRITICAL_RISK_THRESHOLD = 0.7 
PAST_DATA_DIR = "past_data" # Assumed directory for historical files
UPDATE_RATE = 10 # Only update charts and alerts every 10 data points for performance

# --- 1. EXPANDED COLUMN MAPPING (Based on User Inputs) ---
# Map the potentially verbose CSV headers to clean, consistent names.
COL_MAPPING = {
    'ENGINE_RUN_TINE ()': 'timestamp',
    # Engine Performance
    'ENGINE_RPM ()': 'ENGINE_RPM',
    'VEHICLE_SPEED ()': 'VEHICLE_SPEED',
    'ENGINE_LOAD ()': 'ENGINE_LOAD',
    'TIMING_ADVANCE ()': 'TIMING_ADVANCE',
    # Fuel/Air System
    'LONG_TERM_FUEL_TRIM_BANK_1 ()': 'LONG_TERM_FUEL_TRIM_BANK_1',
    'SHORT_TERM_FUEL_TRIM_BANK_1 ()': 'SHORT_TERM_FUEL_TRIM_BANK_1',
    'FUEL_AIR_COMMANDED_EQUIV_RATIO ()': 'FUEL_AIR_EQUIV_RATIO',
    'INTAKE_MANIFOLD_PRESSURE ()': 'INTAKE_MANIFOLD_PRESSURE',
    'INTAKE_AIR_TEMP ()': 'INTAKE_AIR_TEMP',
    # Throttle System
    'THROTTLE ()': 'THROTTLE',
    'ABSOLUTE_THROTTLE_B ()': 'ABSOLUTE_THROTTLE_B',
    'RELATIVE_THROTTLE_POSITION ()': 'RELATIVE_THROTTLE_POSITION',
    'COMMANDED_THROTTLE_ACTUATOR ()': 'COMMANDED_THROTTLE_ACTUATOR',
    # Temperature/Thermal
    'COOLANT_TEMPERATURE ()': 'COOLANT_TEMPERATURE',
    'CATALYST_TEMPERATURE_BANK1_SENSOR1 ()': 'CATALYST_TEMP_B1_S1',
    'CATALYST_TEMPERATURE_BANK1_SENSOR2 ()': 'CATALYST_TEMP_B1_S2',
    # Emissions/Evap/EGR
    'COMMANDED_EVAPORATIVE_PURGE ()': 'COMMANDED_EVAP_PURGE',
    # Electrical
    'CONTROL_MODULE_VOLTAGE ()': 'CONTROL_MODULE_VOLTAGE',
    # Others (for completeness, not strictly used in current logic)
    'FUEL_TANK ()': 'FUEL_TANK',
}

REQUIRED_PIDS = list(COL_MAPPING.values()) # All PIDs are required for comprehensive prediction
REQUIRED_PIDS.remove('timestamp')

KEY_PIDS_DISPLAY = ['ENGINE_RPM', 'COOLANT_TEMPERATURE', 'VEHICLE_SPEED', 'LONG_TERM_FUEL_TRIM_BANK_1', 'CONTROL_MODULE_VOLTAGE', 'CATALYST_TEMP_B1_S1']

# --- HISTORICAL DATA CALCULATION (Fine-Tuning) ---

@st.cache_data
def calculate_historical_stats(data_dir: str) -> Dict[str, Any]:
    """
    Reads all historical drive files and calculates the model's 'training' thresholds 
    for comprehensive fault detection across multiple PIDs.
    """
    all_data = []
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not file_paths:
        st.error(f"❌ **DATA ERROR:** No CSV files found in the local '{data_dir}/' directory.")
        return None

    st.info(f"📚 Fine-tuning prediction model using **{len(file_paths)}** historical drive files from '{data_dir}/'...")

    # Load and clean all historical data using the new mapping
    training_cols = [v for k,v in COL_MAPPING.items() if v != 'timestamp']
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            df = df.rename(columns=COL_MAPPING, errors='ignore')
            
            if all(col in df.columns for col in training_cols):
                all_data.append(df[training_cols].copy())
            else:
                st.warning(f"Skipping {os.path.basename(file_path)}: Missing required columns.")
        except Exception as e:
            st.warning(f"Skipping {os.path.basename(file_path)}: Error reading file ({e}).")

    if not all_data:
        st.error("❌ **DATA ERROR:** All historical files were unreadable or missing required columns. Cannot fine-tune.")
        return None

    master_df = pd.concat(all_data, ignore_index=True)
    
    # --- FIX FOR TypeError: '<' not supported between instances of 'float' and 'str' ---
    # Coerce all training columns to numeric types. Any non-numeric value (like a stray string)
    # will be converted to NaN, allowing quantile calculation to proceed correctly.
    for col in training_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    # --- END FIX ---

    # --- Calculation Logic: Determine 95th percentile for normal operation ---
    stats = {}
    
    # Thermal Thresholds
    stats["COOLANT_TEMP_NORMAL_MAX"] = master_df['COOLANT_TEMPERATURE'].quantile(0.95)
    stats["COOLANT_TEMP_FAULT_MIN"] = stats["COOLANT_TEMP_NORMAL_MAX"] + 6.0 
    stats["COOLANT_TEMP_CRITICAL_FAIL"] = stats["COOLANT_TEMP_FAULT_MIN"] + 7.0 
    
    # Fuel Trim Thresholds
    abs_trim = master_df[['LONG_TERM_FUEL_TRIM_BANK_1', 'SHORT_TERM_FUEL_TRIM_BANK_1']].abs().max(axis=1)
    stats["FUEL_TRIM_NORMAL_DEVIATION"] = abs_trim.quantile(0.95)
    stats["FUEL_TRIM_FAULT_THRESHOLD"] = stats["FUEL_TRIM_NORMAL_DEVIATION"] * 2.0 # Lean/Rich threshold (e.g., > 10% is fault)

    # Electrical Thresholds
    stats["VOLTAGE_NORMAL_MIN"] = master_df['CONTROL_MODULE_VOLTAGE'].quantile(0.05) 
    stats["VOLTAGE_CRITICAL_MIN"] = stats["VOLTAGE_NORMAL_MIN"] * 0.95 # e.g., if normal min is 13.5V, critical is ~12.8V
    
    # Emission/Cat Thresholds
    cat_delta = (master_df['CATALYST_TEMP_B1_S1'] - master_df['CATALYST_TEMP_B1_S2']).abs()
    stats["CAT_DEGRADATION_MAX_DELTA"] = cat_delta.quantile(0.90) + 50 # Allow a safe delta, fault if much higher or lower
    
    # Other Constants
    # Use .min() to check if vehicle speed is available before filtering
    if 'VEHICLE_SPEED' in master_df.columns and master_df['VEHICLE_SPEED'].dropna().min() is not np.nan:
        idle_data = master_df[master_df['VEHICLE_SPEED'] < 5]
        stats["IDLE_IMAP_MAX"] = idle_data['INTAKE_MANIFOLD_PRESSURE'].quantile(0.95)
        stats["IDLE_RPM_MAX"] = idle_data['ENGINE_RPM'].quantile(0.95)
    else:
        # Fallback values if speed data is missing or entirely NaN
        stats["IDLE_IMAP_MAX"] = 35.0 
        stats["IDLE_RPM_MAX"] = 900.0

    stats["RISK_DECAY_RATE"] = 0.05

    calculated_stats = stats
    
    st.success("✅ Model fine-tuning complete! Risk thresholds dynamically set.")
    return calculated_stats

# --- GLOBAL MODEL INIT (Runs once on app startup) ---
try:
    HISTORICAL_STATS = calculate_historical_stats(PAST_DATA_DIR)
except NameError:
    # This block is here for initial execution flow, but the main error is handled above
    HISTORICAL_STATS = None


# --- 2. CORE PREDICTIVE MODEL LOGIC (Includes all 8 Fault Categories) ---

def apply_digital_twin_logic(df, historical_stats: Dict[str, Any], new_file_name: str):
    """
    Applies the prediction logic, incorporating all 8 fault categories from user requirements.
    """
    st.info(f"Processing new file: **{new_file_name}**. Starting comprehensive prediction...")

    df = df.rename(columns=COL_MAPPING, errors='ignore')
    
    if not all(col in df.columns for col in REQUIRED_PIDS):
        missing_cols = [c for c in REQUIRED_PIDS if c not in df.columns]
        st.error(f"The uploaded file is missing required columns: {missing_cols}. Please check headers.")
        return None
    
    # Ensure uploaded file data is also numeric for consistency and safety
    for col in REQUIRED_PIDS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df[list(COL_MAPPING.values())].copy()
    
    # Initialize prediction columns
    df['Prediction_Risk_Score'] = 0.0
    df['Prediction_Risk_Score_Accumulated'] = 0.0 
    df['Predicted_COOLANT_TEMP'] = df['COOLANT_TEMPERATURE'] 
    df['Predicted_FUEL_TRIM'] = df['LONG_TERM_FUEL_TRIM_BANK_1'] 
    df['Predicted_Fault_Type'] = 'None'
    df['Justification'] = 'Normal operation according to fine-tuned model.'
    df['Fault_Category'] = 'Normal'

    # Apply Prediction Rules using Historical Stats and Risk Memory
    for i in range(len(df)):
        row = df.iloc[i]
        
        # --- A. Calculate Instant Risk from 8 Categories ---
        max_instant_risk = 0.0
        current_fault_type = 'None'
        current_justification = 'Normal operation according to fine-tuned model.'
        current_fault_category = 'Normal'

        risk_scores = {}
        
        # 1. Cooling & Thermal System Faults (High Priority)
        temp_risk = 0.0
        if row['COOLANT_TEMPERATURE'] > historical_stats["COOLANT_TEMP_FAULT_MIN"]:
            temp_risk = np.clip(
                (row['COOLANT_TEMPERATURE'] - historical_stats["COOLANT_TEMP_FAULT_MIN"]) / 
                (historical_stats["COOLANT_TEMP_CRITICAL_FAIL"] - historical_stats["COOLANT_TEMP_FAULT_MIN"]), 
                0.0, 1.0
            )
        risk_scores['Thermal'] = (temp_risk, "Overheating Prediction", f"Coolant Temp rising rapidly ({row['COOLANT_TEMPERATURE']:.1f}°C) approaching critical failure threshold.")


        # 2. Fuel & Air Mixture Faults (High Priority)
        trim_risk = 0.0
        abs_ltft = np.abs(row['LONG_TERM_FUEL_TRIM_BANK_1'])
        abs_stft = np.abs(row['SHORT_TERM_FUEL_TRIM_BANK_1'])
        justification = "N/A"
        
        if abs_ltft > historical_stats["FUEL_TRIM_FAULT_THRESHOLD"] or abs_stft > historical_stats["FUEL_TRIM_FAULT_THRESHOLD"]:
             trim_risk = np.clip(
                (max(abs_ltft, abs_stft) - historical_stats["FUEL_TRIM_FAULT_THRESHOLD"]) / 
                (historical_stats["FUEL_TRIM_FAULT_THRESHOLD"] * 0.5), # 50% buffer to hit 1.0 risk
                0.0, 1.0
            )
             if row['LONG_TERM_FUEL_TRIM_BANK_1'] > historical_stats["FUEL_TRIM_FAULT_THRESHOLD"]:
                 justification = f"Persistent Lean Mixture (LTFT: +{abs_ltft:.1f}%) detected; possible injector clogging or vacuum leak."
             else:
                 justification = f"Persistent Rich Mixture (LTFT: -{abs_ltft:.1f}%) detected; possible fuel pressure regulator fault."
        
        # Vacuum Leak check at idle
        if row['VEHICLE_SPEED'] < 5 and row['INTAKE_MANIFOLD_PRESSURE'] > historical_stats["IDLE_IMAP_MAX"] and abs_stft > 10.0:
            trim_risk = max(trim_risk, 0.6) # Elevated Medium Risk for vacuum leak
            justification = f"Vacuum Leak signature: High IMAP ({row['INTAKE_MANIFOLD_PRESSURE']:.0f} kPa) at idle with high STFT."

        risk_scores['Fuel/Air'] = (trim_risk, "Fuel Mixture Anomaly", justification if trim_risk > 0 else "N/A")

        
        # 3. Electrical & Power Delivery Faults
        voltage_risk = 0.0
        if row['CONTROL_MODULE_VOLTAGE'] < historical_stats["VOLTAGE_NORMAL_MIN"]:
            voltage_risk = np.clip(
                (historical_stats["VOLTAGE_NORMAL_MIN"] - row['CONTROL_MODULE_VOLTAGE']) / 
                (historical_stats["VOLTAGE_NORMAL_MIN"] - historical_stats["VOLTAGE_CRITICAL_MIN"]), 
                0.0, 0.7
            ) # Cap at 0.7 (Critical Alert)
            if row['CONTROL_MODULE_VOLTAGE'] < historical_stats["VOLTAGE_CRITICAL_MIN"]:
                 voltage_fault = "Alternator/Battery Critical Failure"
            else:
                 voltage_fault = "Weak Battery Early Sign"
            risk_scores['Electrical'] = (voltage_risk, voltage_fault, f"Control Module Voltage is low ({row['CONTROL_MODULE_VOLTAGE']:.2f}V), predicting power failure.")

        
        # 4. Emission & Catalyst Faults
        cat_risk = 0.0
        cat_delta_actual = np.abs(row['CATALYST_TEMP_B1_S1'] - row['CATALYST_TEMP_B1_S2'])
        if cat_delta_actual > historical_stats["CAT_DEGRADATION_MAX_DELTA"]:
             cat_risk = np.clip(
                (cat_delta_actual - historical_stats["CAT_DEGRADATION_MAX_DELTA"]) / 
                (historical_stats["CAT_DEGRADATION_MAX_DELTA"] * 0.5),
                0.0, 0.6
             ) # Max risk 0.6 (Elevated Risk)
             risk_scores['Emission'] = (cat_risk, "Catalytic Converter Degradation", f"Catalyst Temp Delta is abnormally high ({cat_delta_actual:.0f}C), indicating efficiency loss.")
             
        
        # 5. Engine Performance Faults
        perf_risk = 0.0
        # Simple Logic: High load/Low speed (Torque Loss)
        if row['ENGINE_LOAD'] > 80 and row['VEHICLE_SPEED'] < 10 and row['ENGINE_RPM'] > 2000:
            perf_risk = 0.4 # Low risk marker
            risk_scores['Performance'] = (perf_risk, "Poor Power Delivery / Torque Loss", "Engine under max load with minimal speed increase; poor power/torque response.")
        
        # 6. Throttle & Air Intake System Faults
        throttle_risk = 0.0
        throttle_mismatch = np.abs(row['COMMANDED_THROTTLE_ACTUATOR'] - row['THROTTLE'])
        if throttle_mismatch > 10.0: # Check for >10% command/response lag
            throttle_risk = 0.5
            risk_scores['Throttle'] = (throttle_risk, "Throttle Body Carbon Buildup/Lag", f"Commanded throttle ({row['COMMANDED_THROTTLE_ACTUATOR']:.1f}%) is lagging actual position ({row['THROTTLE']:.1f}%) by over 10%.")
            
        
        # 7. Sensor Degradation / Drift Faults (Indirect check via performance metrics)
        # Add a check for idle instability (RPM oscillation when speed=0)
        idle_oscillation_risk = 0.0
        if i > 5 and row['VEHICLE_SPEED'] < 2:
            rpm_std = df.iloc[i-5:i+1]['ENGINE_RPM'].std()
            if rpm_std > 30.0 and row['ENGINE_RPM'] > historical_stats["IDLE_RPM_MAX"]:
                idle_oscillation_risk = 0.5
                risk_scores['Sensor Degradation'] = (idle_oscillation_risk, "Engine Idle Instability", f"High RPM fluctuation ({rpm_std:.1f} RPM standard deviation) at idle, suggesting potential sensor/actuator drift or engine misfire.")
            
        
        # 8. Driver Behavior-Induced Wear (Risk Assessment)
        wear_risk = 0.0
        if row['ENGINE_RPM'] > 5000 and row['ENGINE_LOAD'] > 90:
             wear_risk = 0.3
             risk_scores['Driver Behavior'] = (wear_risk, "Frequent High RPM/Load Wear", "Detected high-stress driving pattern (High RPM/Load) which accelerates mechanical wear.")
        
        # --- Consolidate Max Instant Risk ---
        # Find the highest risk and the corresponding fault
        for category, (risk, fault, justification) in risk_scores.items():
            if risk > max_instant_risk:
                max_instant_risk = risk
                current_fault_type = fault
                current_justification = justification
                current_fault_category = category

        # --- B. Apply Risk Memory (Accumulation and Decay) ---
        previous_risk = df['Prediction_Risk_Score_Accumulated'].iloc[i-1] if i > 0 else 0.0
        decayed_previous_risk = previous_risk * (1 - historical_stats["RISK_DECAY_RATE"])
        
        accumulated_risk = max(max_instant_risk, decayed_previous_risk)
        
        df.loc[df.index[i], 'Prediction_Risk_Score'] = accumulated_risk
        df.loc[df.index[i], 'Prediction_Risk_Score_Accumulated'] = accumulated_risk 

        # Only update fault details if the accumulated risk is above 0.5 or if the instant risk was higher than decayed memory
        if accumulated_risk >= 0.5:
             if max_instant_risk >= accumulated_risk or accumulated_risk > 0.7:
                 df.loc[df.index[i], 'Predicted_Fault_Type'] = current_fault_type
                 df.loc[df.index[i], 'Justification'] = current_justification
                 df.loc[df.index[i], 'Fault_Category'] = current_fault_category
             
             # --- C. Apply Prognosis (Project 90s ahead based on accumulated risk) ---
             if accumulated_risk > CRITICAL_RISK_THRESHOLD:
                # Project Coolant Temp ramp up (Thermal Failure Prognosis)
                projected_temp = row['COOLANT_TEMPERATURE'] + (accumulated_risk * 20)
                df.loc[df.index[i], 'Predicted_COOLANT_TEMP'] = projected_temp
                
                # Project Fuel Trim drift (Mixture Failure Prognosis)
                trim_direction = np.sign(row['LONG_TERM_FUEL_TRIM_BANK_1']) if np.abs(row['LONG_TERM_FUEL_TRIM_BANK_1']) > 0.1 else 1
                projected_trim = row['LONG_TERM_FUEL_TRIM_BANK_1'] + (trim_direction * accumulated_risk * 10)
                df.loc[df.index[i], 'Predicted_FUEL_TRIM'] = np.clip(projected_trim, -30.0, 30.0)

        # Convert 'timestamp' (Engine Run Time) to datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', origin='unix')
    df = df.set_index('timestamp')
    
    return df

# --- 3. NEW FAULT SUMMARY GENERATION ---

def generate_fault_type_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Creates the final, detailed report table based on the 8 fault categories."""
    
    # Fault Category mapping is based on the logic in apply_digital_twin_logic
    fault_categories_user = [
        "Thermal", "Fuel/Air", "Electrical", "Emission", 
        "Performance", "Throttle", "Sensor Degradation", "Driver Behavior"
    ]
    
    summary_data = []
    
    for category in fault_categories_user:
        # Find all points where this category was the dominant or predicted fault type
        fault_rows = data[data['Fault_Category'] == category]
        
        if fault_rows.empty:
            max_risk = 0.0
            highest_risk_type = "None Detected"
            justification = "System remained within historical safe bounds for this category."
        else:
            max_risk = fault_rows['Prediction_Risk_Score'].max()
            # Get the predicted fault and justification at the MAX risk point
            max_row_index = fault_rows['Prediction_Risk_Score'].idxmax()
            max_row = fault_rows.loc[max_row_index]
            
            highest_risk_type = max_row['Predicted_Fault_Type']
            justification = max_row['Justification']
        
        summary_data.append({
            "Fault Category": category,
            "Max Risk Score (%)": f"{max_risk:.1%}",
            "Highest Predicted Fault Type": highest_risk_type,
            "Justification / Key Indicator": justification
        })
        
    return pd.DataFrame(summary_data)

# --- Remaining Functions (calculate_single_kpis, create_charts, etc. - UNCHANGED) ---

def calculate_single_kpis(df, historical_stats):
    """Calculates key performance indicators for a single drive file."""
    
    max_risk = df['Prediction_Risk_Score'].max()
    critical_alerts = (df['Prediction_Risk_Score'] >= CRITICAL_RISK_THRESHOLD).sum()

    lead_time_sec = np.nan
    
    critical_alert_rows = df[df['Prediction_Risk_Score'] >= CRITICAL_RISK_THRESHOLD]
    first_alert_time = critical_alert_rows.index.min().timestamp() if not critical_alert_rows.empty else None
        
    failure_rows = df[df['COOLANT_TEMPERATURE'] >= historical_stats["COOLANT_TEMP_CRITICAL_FAIL"]]
    first_failure_time = failure_rows.index.min().timestamp() if not failure_rows.empty else None

    if first_alert_time is not None and first_failure_time is not None:
        if first_alert_time < first_failure_time:
             lead_time_sec = first_failure_time - first_alert_time
        else:
             lead_time_sec = -1 
    elif first_alert_time is None and first_failure_time is not None:
         lead_time_sec = -1 

    def format_lead_time(x):
        if pd.isna(x):
            return "N/A (No Alerts/Failure)"
        elif x == -1:
            return "FAILED (Alert After Failure)"
        else:
            return f"{x:.0f}s"
            
    return {
        'Max Risk Score': f"{max_risk:.2f}",
        'Critical Alerts (#)': critical_alerts,
        'Prognostic Lead Time': format_lead_time(lead_time_sec)
    }

def create_temp_chart(rolling_data, historical_stats, current_data):
    FUTURE_PREDICTION_HORIZON_S = 90
    future_timestamp = current_data.name + pd.Timedelta(seconds=FUTURE_PREDICTION_HORIZON_S)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_data.index, y=rolling_data['COOLANT_TEMPERATURE'], 
        mode='lines', name='Actual Temp', line=dict(color='blue', width=2)
    ))
    
    forecast_df = pd.DataFrame({
        'timestamp': [current_data.name, future_timestamp],
        'value': [current_data['COOLANT_TEMPERATURE'], current_data['Predicted_COOLANT_TEMP']]
    }).set_index('timestamp')
    
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['value'], 
        mode='lines+markers', name=f'Predicted ({FUTURE_PREDICTION_HORIZON_S}s Ahead)',
        line=dict(color='red', dash='dash', width=2),
        marker=dict(size=10, symbol='circle-open', color='red')
    ))
    
    fig.add_hline(y=historical_stats["COOLANT_TEMP_CRITICAL_FAIL"], line_dash="dot", 
                  annotation_text=f"Failure Threshold ({historical_stats['COOLANT_TEMP_CRITICAL_FAIL']:.0f}°C)", 
                  annotation_position="bottom right", line_color="orange")

    fig.update_layout(height=350, title="Engine Coolant Temperature (Actual vs. Projected)", margin=dict(t=50, b=0), showlegend=True, hovermode="x unified")
    return fig

def create_trim_chart(rolling_data, historical_stats, current_data):
    FUTURE_PREDICTION_HORIZON_S = 90
    future_timestamp = current_data.name + pd.Timedelta(seconds=FUTURE_PREDICTION_HORIZON_S)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_data.index, y=rolling_data['LONG_TERM_FUEL_TRIM_BANK_1'], 
        mode='lines', name='Actual LTFT (%)', line=dict(color='green', width=2)
    ))
    
    forecast_df = pd.DataFrame({
        'timestamp': [current_data.name, future_timestamp],
        'value': [current_data['LONG_TERM_FUEL_TRIM_BANK_1'], current_data['Predicted_FUEL_TRIM']]
    }).set_index('timestamp')
    
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['value'], 
        mode='lines+markers', name=f'Predicted Drift ({FUTURE_PREDICTION_HORIZON_S}s Ahead)',
        line=dict(color='red', dash='dash', width=2),
        marker=dict(size=10, symbol='circle-open', color='red')
    ))
    
    fig.add_hline(y=historical_stats["FUEL_TRIM_FAULT_THRESHOLD"], line_dash="dot", 
                  annotation_text=f"Fault Threshold (+{historical_stats['FUEL_TRIM_FAULT_THRESHOLD']:.1f}%)", 
                  annotation_position="top right", line_color="orange")
    fig.add_hline(y=-historical_stats["FUEL_TRIM_FAULT_THRESHOLD"], line_dash="dot", 
                  annotation_text=f"Fault Threshold (-{historical_stats['FUEL_TRIM_FAULT_THRESHOLD']:.1f}%)", 
                  annotation_position="bottom right", line_color="orange")

    fig.update_layout(height=350, title="Long Term Fuel Trim (LTFT) - Actual vs. Projected Drift", margin=dict(t=50, b=0), showlegend=True, hovermode="x unified")
    return fig

def create_risk_chart(rolling_data, critical_risk_threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_data.index, y=rolling_data['Prediction_Risk_Score'], 
        mode='lines', name='Prediction Risk Score', line=dict(color='purple', width=3)
    ))
    fig.add_hline(y=critical_risk_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"CRITICAL Threshold ({critical_risk_threshold:.0%})",
                  annotation_position="top left")
    
    fig.update_layout(height=350, title="System Health Risk Score Over Time", margin=dict(t=50, b=0), showlegend=True, hovermode="x unified",
                      yaxis=dict(range=[0, 1.0]))
    return fig


# --- 4. STREAMLIT DASHBOARD LAYOUT (Updated to include new summary table) ---

st.set_page_config(layout="wide")
st.title("🚀 Max-Speed Digital Twin: Comprehensive Prognostic Report Generation")
st.markdown("---")

# --- HISTORICAL DATA STATUS & INIT CHECK ---
if HISTORICAL_STATS is None:
    st.error(f"Please ensure you have created a folder named **`{PAST_DATA_DIR}`** in the same directory as this script and placed all your historical CSV files inside it. Then restart the app.")
    st.stop()
    
st.sidebar.subheader("Fine-Tuned Model Parameters (from Local Database)")
st.sidebar.json(HISTORICAL_STATS)


# --- STEP 1: UPLOAD NEW FILE FOR PREDICTION ---
st.header("Step 1: Upload New File for Prediction")
st.info("The prediction model has been fine-tuned using the historical data found in your local `past_data/` folder and is now using all 19 available PIDs for comprehensive fault checks.")

uploaded_file_new = st.file_uploader(
    "Upload the **NEW** Raw Drive CSV File you want to predict on.",
    type="csv",
    key="new_file_uploader"
)

if uploaded_file_new is None:
    st.warning("Please upload the new CSV file you wish to analyze.")
    st.stop()


# --- STEP 2: RUN PREDICTION AND SIMULATION ---
try:
    start_time = time.time()
    raw_df_new = pd.read_csv(uploaded_file_new)
    data = apply_digital_twin_logic(raw_df_new, HISTORICAL_STATS, uploaded_file_new.name)
    
    if data is None:
        st.stop()

except Exception as e:
    st.error(f"An error occurred during new file processing: {e}")
    st.stop()


# --- DYNAMIC SIMULATION SETUP ---
st.markdown("---")
st.header(f"Step 2: Max-Speed Simulation: `{uploaded_file_new.name}`")
st.warning("The simulation runs at **maximum speed** to ensure the final report is generated in under 5 seconds.")

# Placeholders for dynamic updates
header_placeholder = st.empty()
alerts_placeholder = st.empty()
chart_col1_placeholder = st.empty() 
chart_col2_placeholder = st.empty() 
progress_bar = st.progress(0)

# --- REAL-TIME SIMULATION LOOP (No Delay) ---

for i in range(len(data)):
    
    current_data = data.iloc[i]
    
    # Update Charts and Alerts only every 'UPDATE_RATE' iterations to keep the browser responsive
    if i % UPDATE_RATE == 0 or i == len(data) - 1:
        
        rolling_data = data.iloc[max(0, i - WINDOW_SIZE):i + 1].copy()

        # 2. Display Metrics (Current Status PIDs)
        with header_placeholder.container():
            st.markdown(f"### ⏱️ Live Data Stream | Current Time: **{current_data.name.strftime('%H:%M:%S')}**")
            cols = st.columns(len(KEY_PIDS_DISPLAY))
            for k, pid in enumerate(KEY_PIDS_DISPLAY):
                cols[k].metric(pid, f"{current_data[pid]:.1f}")
                
        # 3. Prediction Visualization
        with chart_col1_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_temp_chart(rolling_data, HISTORICAL_STATS, current_data), use_container_width=True)
            with col2:
                st.plotly_chart(create_trim_chart(rolling_data, HISTORICAL_STATS, current_data), use_container_width=True)
                
        with chart_col2_placeholder.container():
            st.plotly_chart(create_risk_chart(rolling_data, CRITICAL_RISK_THRESHOLD), use_container_width=True)


        # 4. Alerting & Explainability (Faliers/Issues)
        risk_score = current_data['Prediction_Risk_Score']
        
        if risk_score >= CRITICAL_RISK_THRESHOLD:
            with alerts_placeholder.container():
                st.error(f"## 🚨 CRITICAL ALERT: {current_data['Predicted_Fault_Type']}")
                st.markdown(f"**Confidence:** **{risk_score:.0%}** | **Fault Category:** **{current_data['Fault_Category']}** | **Justification (Fine-Tuned):** _{current_data['Justification']}_")
                
        elif risk_score >= 0.5:
            with alerts_placeholder.container():
                st.warning(f"## ⚠️ ELEVATED RISK: {current_data['Predicted_Fault_Type']}")
                st.markdown(f"**Confidence:** **{risk_score:.0%}** | **Fault Category:** **{current_data['Fault_Category']}** | **Justification (Fine-Tuned):** _{current_data['Justification']}_")

        else:
            alerts_placeholder.empty()

    # Update progress bar every iteration
    progress_bar.progress((i + 1) / len(data))
    
end_time = time.time()
processing_time = end_time - start_time
    
progress_bar.empty()
st.success(f"✅ **Report Generation Complete!** Processing time: **{processing_time:.2f} seconds**.")


# --- 5. FINAL PROGNOSIS DISPLAY (Current Status & Next Near Status) ---
st.markdown("---")
st.header("Step 3: Prediction Output Report")

final_data = data.iloc[-1]
final_risk = final_data['Prediction_Risk_Score']
final_temp_actual = final_data['COOLANT_TEMPERATURE']
final_trim_actual = final_data['LONG_TERM_FUEL_TRIM_BANK_1']

final_temp_pred = final_data['Predicted_COOLANT_TEMP']
final_trim_pred = final_data['Predicted_FUEL_TRIM']
final_fault = final_data['Predicted_Fault_Type']


if final_risk >= CRITICAL_RISK_THRESHOLD:
    box_color = '#dc3545'
    alert_text = "🚨 CRITICAL FAULT PREDICTED"
elif final_risk >= 0.5:
    box_color = '#ffc107'
    alert_text = "⚠️ ELEVATED RISK PREDICTED"
else:
    box_color = '#28a745'
    alert_text = "✅ NORMAL OPERATION"

st.markdown(f"""
<div style="padding: 15px; border-radius: 10px; border: 2px solid {box_color}; background-color: rgba(255, 255, 255, 0.05);">
    <h3 style="color: {box_color};">{alert_text} | Max Risk: {final_risk:.1%}</h3>
    <div style="display: flex; justify-content: space-around; text-align: center;">
        <div>
            <h4>Current Status (End of Test)</h4>
            <p style="font-size: 16px; margin-bottom: 0px;">Coolant Temp: <strong>{final_temp_actual:.2f}°C</strong></p>
            <p style="font-size: 16px; margin-top: 5px;">Fuel Trim: <strong>{final_trim_actual:.2f}%</strong></p>
        </div>
        <div>
            <h4>Next Near Status (90s Prognosis)</h4>
            <p style="font-size: 16px; margin-bottom: 0px; color: {'red' if final_risk > 0.7 else 'inherit'};">Projected Temp: <strong>{final_temp_pred:.2f}°C</strong></p>
            <p style="font-size: 16px; margin-top: 5px; color: {'red' if final_risk > 0.7 else 'inherit'};">Projected Trim: <strong>{final_trim_pred:.2f}%</strong></p>
        </div>
    </div>
    <hr style="border-top: 1px solid #ccc;"/>
    <p><strong>Dominant Fault Category:</strong> {final_data['Fault_Category']}</p>
    <p><strong>Predicted Fault Type:</strong> {final_fault}</p>
    <p><strong>Justification:</strong> <em>{final_data['Justification']}</em></p>
</div>
""", unsafe_allow_html=True)


# --- 6. DETAILED FAULT CAPABILITY REPORT (New Requirement) ---
st.markdown("---")
st.header("Step 4: OBD-Based Fault Detection Capability Report (Across All Categories)")

fault_summary_df = generate_fault_type_summary(data)
st.dataframe(fault_summary_df, use_container_width=True, hide_index=True)


# --- 7. PERFORMANCE METRICS (Failures, Risk, Issue Details) ---
st.subheader("Prognostic Performance Summary")
kpi_results = calculate_single_kpis(data, HISTORICAL_STATS)

kpi_cols = st.columns(3)
kpi_cols[0].metric("Maximum Risk Score", kpi_results['Max Risk Score'], help="The highest risk value recorded during the drive.")
kpi_cols[1].metric("Critical Alerts Issued", kpi_results['Critical Alerts (#)'], help="Number of data points where the risk exceeded the 70% CRITICAL threshold.")
kpi_cols[2].metric("Prognostic Lead Time", kpi_results['Prognostic Lead Time'], help="Time difference between the first CRITICAL ALERT (0.7) and the actual physical failure threshold breach (if applicable).")