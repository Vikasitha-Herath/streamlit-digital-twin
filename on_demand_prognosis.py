import numpy as np
import pandas as pd

# --- HYBRID TWIN MODEL CONSTANTS (Validated) ---
K1_HEAT_GENERATION = 0.0000005
K2_HEAT_DISSIPATION = 0.000001
K3_STATIC_HEAT_LOSS = 0.00015
FAULT_TEMP_THRESHOLD = 105.0 

# --- CURRENT SENSOR DATA INPUT ---
# This dictionary represents the data transmitted by the vehicle's ECU right now.
# (This sample data is modeled after a high-risk point in your simulated drive.)
CURRENT_DATA_SNAPSHOT = {
    'ENGINE_LOAD': 45.0, 
    'ENGINE_RPM': 2100.0, 
    'COOLANT_TEMPERATURE': 103.5, # High current temperature
    'VEHICLE_SPEED': 75.0, 
    'LONG_TERM_FUEL_TRIM_BANK_1': -1.0, 
    'ENGINE_RUN_TINE': 1200.0 # Time proxy for fault severity
}

# --- 1. CORE PREDICTION FUNCTIONS ---

def physics_based_model(data_point):
    """Calculates the 90s temperature prediction based on simplified thermal physics."""
    load = data_point['ENGINE_LOAD']
    rpm = data_point['ENGINE_RPM']
    temp = data_point['COOLANT_TEMPERATURE']
    speed = data_point['VEHICLE_SPEED']
    
    heat_in = K1_HEAT_GENERATION * (load * rpm)
    heat_out_dynamic = K2_HEAT_DISSIPATION * (temp * speed)
    heat_out_static = K3_STATIC_HEAT_LOSS * temp

    # 90.0 is the prediction time horizon in seconds
    delta_t = (heat_in - heat_out_dynamic - heat_out_static) * 90.0
    
    predicted_temp = temp + delta_t
    return max(predicted_temp, 30.0)

def residual_ml_model_predict(data_point):
    """Simulates the ML model predicting the residual error and risk score."""
    base_residual = 0.0 
    risk_score = 0.05
    fault_type = "Normal Operation"
    justification = "System operating within normal physical parameters."
    
    # We are simulating the "Overheating" fault based on time and temp
    time_proxy = data_point['ENGINE_RUN_TINE']
    temp = data_point['COOLANT_TEMPERATURE']
    
    # Logic to inject fault based on high current temperature (ML detects the pattern)
    if temp >= 100.0:
        ml_residual_push = 4.0 + (time_proxy / 1000.0) * 0.02
        base_residual = ml_residual_push # ML pushes the prediction up
        risk_score = min(1.0, 0.5 + base_residual * 0.15)
        fault_type = "Imminent Overheating"
        justification = f"High positive thermal residual ({base_residual:.2f}°C) detected by ML model, suggesting cooling system anomaly."

    risk_score = max(0.0, risk_score)
    return base_residual, risk_score, fault_type, justification

def hybrid_twin_prediction(data_point):
    """Combines PBM and MLM for the final prediction."""
    
    # 1. Run Physics Model (PBM)
    pbm_temp = physics_based_model(data_point)
    
    # 2. Run ML Residual Model (MLM)
    ml_residual, risk_score, fault_type, justification = residual_ml_model_predict(data_point)
    
    # 3. Combine: Final Predicted Temp = PBM Temp + ML Residual
    final_predicted_temp = pbm_temp + ml_residual

    # 4. Refine Risk Score based on threshold crossing
    if final_predicted_temp >= FAULT_TEMP_THRESHOLD:
        risk_score = max(risk_score, 0.9)
        if "Overheating" not in fault_type:
            fault_type = "High Temperature Risk (PBM Alert)"
            
    return final_predicted_temp, risk_score, fault_type, justification

# --- 2. EXECUTION ---
if __name__ == '__main__':
    
    # Calculate the prognosis instantly
    predicted_temp, risk, fault, justification = hybrid_twin_prediction(CURRENT_DATA_SNAPSHOT)

    print("--- 🧠 INSTANT DIGITAL TWIN PROGNOSIS (90 Seconds Ahead) ---")
    print(f"Current Coolant Temperature: {CURRENT_DATA_SNAPSHOT['COOLANT_TEMPERATURE']}°C")
    print("-" * 50)
    
    # Display the final prediction
    if risk >= 0.9:
        print(f"## 🚨 CRITICAL ALERT DETECTED ({fault})")
    elif risk >= 0.7:
        print(f"## ⚠️ HIGH RISK DETECTED ({fault})")
    else:
        print(f"## ✅ Normal Prognosis")
        
    print(f"\nPredicted Temperature (90s): **{predicted_temp:.2f}°C**")
    print(f"Critical Threshold:          **{FAULT_TEMP_THRESHOLD:.2f}°C**")
    print(f"Prediction Risk Score:       **{risk:.2f}**")
    print(f"Justification: {justification}")
    print("-" * 50)