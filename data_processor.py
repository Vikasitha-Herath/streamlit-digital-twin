import pandas as pd
import numpy as np
import os

# --- Configuration for Loading and Saving ---
ORIGINAL_FILES = {
    1: "drive1.csv",
    2: "drive2.csv",
    3: "drive3.csv",
    4: "drive4.csv",
    5: "drive5.csv",
    6: "drive6.csv",
    7: "drive7.csv",
}
OUTPUT_DIR = "data"

# --- Hybrid Twin Model Constants (Recalibrated from previous step) ---
# These constants simulate the physics of engine cooling and heating
K1_HEAT_GENERATION = 0.0000005
K2_HEAT_DISSIPATION = 0.000001
K3_STATIC_HEAT_LOSS = 0.00015
FAULT_TEMP_THRESHOLD = 105.0 

# --- Helper Functions (The Hybrid Twin Logic) ---

def clean_column_names(df):
    """Cleans column names by removing parentheses and extra spaces."""
    df.columns = [col.replace(' ()', '').strip() for col in df.columns]
    return df

def physics_based_model(row):
    """Calculates the 90s temperature prediction based on simplified thermal physics."""
    load = row['ENGINE_LOAD']
    rpm = row['ENGINE_RPM']
    temp = row['COOLANT_TEMPERATURE']
    speed = row['VEHICLE_SPEED']
    
    heat_in = K1_HEAT_GENERATION * (load * rpm)
    heat_out_dynamic = K2_HEAT_DISSIPATION * (temp * speed)
    heat_out_static = K3_STATIC_HEAT_LOSS * temp

    # 90.0 is the prediction time horizon in seconds
    delta_t = (heat_in - heat_out_dynamic - heat_out_static) * 90.0
    
    predicted_temp = temp + delta_t
    return max(predicted_temp, 30.0)

def residual_ml_model_predict(row, scenario_type):
    """Simulates the ML model predicting the residual error and risk score."""
    base_residual = 0.0 
    risk_score = 0.05
    fault_type = "Normal Operation"
    justification = "System operating within normal physical parameters."
    
    time_proxy = row['ENGINE_RUN_TINE']
    ft_long = row['LONG_TERM_FUEL_TRIM_BANK_1']
    
    # --- Simulated Fault Injection Logic based on Scenario ---
    if scenario_type == 'Overheating_2' or scenario_type == 'Overheating_6': 
        ml_residual_push = 3.0 + (time_proxy / 1000.0) * 0.015
        base_residual = ml_residual_push
        risk_score = min(1.0, 0.4 + base_residual * 0.15)
        fault_type = "Imminent Overheating"
        justification = f"High positive thermal residual ({base_residual:.2f}°C) suggests internal cooling failure."
        
    elif scenario_type == 'Sensor Drift_3': 
        if ft_long < -5.0 or ft_long > 5.0:
            base_residual = 0.5 
            risk_score = min(1.0, 0.6 + abs(ft_long) * 0.05)
            fault_type = "Sensor Drift (O2/Fuel Trim)"
            justification = f"Severe long-term fuel trim deviation ({ft_long:.2f}%) detected, indicating possible O2 sensor drift."

    risk_score = max(0.0, risk_score)
    return base_residual, risk_score, fault_type, justification

def hybrid_twin_prediction(df, scenario_name):
    """Applies the hybrid twin logic across the entire DataFrame."""
    
    # 1. Apply PBM
    df['Predicted_PBM_TEMP'] = df.apply(physics_based_model, axis=1)
    
    # 2. Apply ML Residual Model
    results = df.apply(lambda row: residual_ml_model_predict(row, scenario_name), axis=1, result_type='expand')
    results.columns = ['ML_Residual', 'Prediction_Risk_Score', 'Predicted_Fault_Type', 'Justification']
    df = pd.concat([df, results], axis=1)
    
    # 3. Combine: Final Predicted Temp = PBM Temp + ML Residual
    df['Predicted_COOLANT_TEMP'] = df['Predicted_PBM_TEMP'] + df['ML_Residual']

    # 4. Refine Risk Score based on threshold crossing and select columns
    df['Prediction_Risk_Score'] = np.where(
        df['Predicted_COOLANT_TEMP'] >= FAULT_TEMP_THRESHOLD,
        np.maximum(df['Prediction_Risk_Score'], 0.8),
        df['Prediction_Risk_Score']
    )
    df['Predicted_Fault_Type'] = np.where(
        (df['Predicted_COOLANT_TEMP'] >= FAULT_TEMP_THRESHOLD) & (df['Predicted_Fault_Type'] == "Normal Operation"),
        "High Temperature Risk (PBM Alert)",
        df['Predicted_Fault_Type']
    )
    df['timestamp'] = df['ENGINE_RUN_TINE']
                  
    return df[['timestamp', 'ENGINE_RPM', 'VEHICLE_SPEED', 'COOLANT_TEMPERATURE', 'ENGINE_LOAD', 
               'LONG_TERM_FUEL_TRIM_BANK_1', 'Predicted_COOLANT_TEMP', 'Prediction_Risk_Score', 
               'Predicted_Fault_Type', 'Justification']]


# --- Main Execution ---

def process_all_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Starting data processing and Hybrid Twin simulation...")

    for num, filename in ORIGINAL_FILES.items():
        print(f"Processing {filename}...")
        
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"ERROR: Original file '{filename}' not found. Ensure it is in the same directory.")
            continue
            
        # 1. Cleaning
        df = clean_column_names(df)
        
        # 2. Determine Scenario Type for fault injection
        scenario_key = f"drive{num}"
        scenario_map = {'drive2': 'Overheating_2', 'drive3': 'Sensor Drift_3', 'drive6': 'Overheating_6'}
        scenario_type = scenario_map.get(scenario_key, 'Normal')
        
        # 3. Apply Hybrid Twin Prediction
        df_processed = hybrid_twin_prediction(df, scenario_type)
        
        # 4. Save the simulated file
        output_filename = os.path.join(OUTPUT_DIR, f"simulated_{filename}")
        df_processed.to_csv(output_filename, index=False)
        print(f"Successfully saved simulated data to {output_filename}")

    print("\n--- ALL DATA PROCESSING COMPLETE ---")
    print(f"Processed files saved to the '{OUTPUT_DIR}' directory.")
    
if __name__ == '__main__':
    process_all_files()