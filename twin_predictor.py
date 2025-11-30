import pandas as pd
import numpy as np

# --- 1. SIMULATED MODEL PARAMETERS (RECALIBRATED) ---

# REDUCED CONSTANTS: significantly lowered to prevent thermal explosion/crash
K1_HEAT_GENERATION = 0.0000005 # Drastically reduced
K2_HEAT_DISSIPATION = 0.000001 # Reduced
K3_STATIC_HEAT_LOSS = 0.00015 # New constant for natural heat loss independent of speed/temp

# Thresholds
FAULT_TEMP_THRESHOLD = 105.0 

# --- 2. CORE PREDICTION FUNCTIONS ---

def physics_based_model(row):
    """
    Simulates the Physics-Based Model (PBM) for Coolant Temperature forecast.
    Predicts the temperature 90s ahead based on current state and a simplified thermal model.
    """
    
    # Inputs: Load, RPM, Current Temp, Vehicle Speed
    load = row['ENGINE_LOAD']
    rpm = row['ENGINE_RPM']
    temp = row['COOLANT_TEMPERATURE']
    speed = row['VEHICLE_SPEED']
    
    # New calculation: Temp change is heat in - heat out (dissipation + static loss)
    heat_in = K1_HEAT_GENERATION * (load * rpm)
    
    # Dissipation factor: higher speed/temp = more cooling
    heat_out_dynamic = K2_HEAT_DISSIPATION * (temp * speed)
    
    # Static factor: always some cooling due to environment, etc.
    heat_out_static = K3_STATIC_HEAT_LOSS * temp

    delta_t = (heat_in - heat_out_dynamic - heat_out_static) * 90.0 # Multiply by time horizon (90s)
    
    predicted_temp = temp + delta_t
    return predicted_temp

def residual_ml_model_predict(row, scenario_type):
    """
    Simulates the Machine Learning Model (MLM) that predicts the residual and risk score.
    """
    
    base_residual = 0.0 
    risk_score = 0.05
    fault_type = "Normal Operation"
    justification = "System operating within normal physical parameters."

    # --- SIMULATED FAULT INJECTION LOGIC ---
    if scenario_type == 'Overheating':
        # Simulated residual growth (e.g., pump failure)
        # We ensure the ML model pushes the PBM result higher
        ml_residual_push = 3.0 + (row['timestamp'] / 1000.0) * 0.015
        base_residual = ml_residual_push
        risk_score = min(1.0, 0.4 + base_residual * 0.15)
        fault_type = "Imminent Overheating"
        justification = f"High positive thermal residual ({base_residual:.2f}°C) suggests internal cooling failure."
        
    elif scenario_type == 'Sensor Drift':
        # High risk due to non-thermal parameters (e.g., Fuel Trim deviation)
        ft_long = row['LONG_TERM_FUEL_TRIM_BANK_1']
        if ft_long < -5.0 or ft_long > 5.0:
            base_residual = 0.5 # Minimal temp impact
            risk_score = min(1.0, 0.6 + abs(ft_long) * 0.05)
            fault_type = "Sensor Drift (O2/Fuel Trim)"
            justification = f"Severe long-term fuel trim deviation ({ft_long:.2f}%) detected, indicating possible O2 sensor drift."

    # Prevent negative risk scores
    risk_score = max(0.0, risk_score)
    
    return base_residual, risk_score, fault_type, justification

def hybrid_twin_prediction(row, scenario_type):
    """Combines PBM and MLM for the final prediction."""
    
    # 1. Run Physics Model
    pbm_temp = physics_based_model(row)
    
    # 2. Run ML Residual Model
    ml_residual, risk_score, fault_type, justification = residual_ml_model_predict(row, scenario_type)
    
    # 3. Combine: Final Predicted Temp = PBM Temp + ML Residual
    final_predicted_temp = pbm_temp + ml_residual

    # 4. Refine Risk Score based on final predicted temp crossing threshold
    if final_predicted_temp >= FAULT_TEMP_THRESHOLD:
        # Increase risk if the hybrid prediction crosses the critical line
        risk_score = max(risk_score, 0.8)
        if "Overheating" not in fault_type:
            fault_type = "High Temperature Risk (PBM Alert)"

    # Ensure predicted temp is not ridiculously low (e.g., below ambient)
    final_predicted_temp = max(final_predicted_temp, 30.0)
            
    return final_predicted_temp, risk_score, fault_type, justification

# --- 3. EXECUTION EXAMPLE (How this would be used on the server) ---

if __name__ == '__main__':
    print("--- Digital Twin Prediction Engine Test (RECALIBRATED) ---")
    
    # Example Row 1: High RPM, High Load (Simulating a hill climb)
    test_data_normal = {
        'ENGINE_LOAD': 80.0, 'ENGINE_RPM': 3500.0, 
        'COOLANT_TEMPERATURE': 92.0, 'VEHICLE_SPEED': 60.0, 
        'LONG_TERM_FUEL_TRIM_BANK_1': 1.0, 'timestamp': 100
    }
    
    # Example Row 2: Simulating start of a pump failure (Overheating scenario)
    test_data_fault = {
        'ENGINE_LOAD': 40.0, 'ENGINE_RPM': 2000.0, 
        'COOLANT_TEMPERATURE': 102.0, 'VEHICLE_SPEED': 80.0, 
        'LONG_TERM_FUEL_TRIM_BANK_1': -2.0, 'timestamp': 1000
    }
    
    # --- Test 1: Normal Operation ---
    predicted_temp_n, risk_n, fault_n, justification_n = hybrid_twin_prediction(test_data_normal, 'Normal')
    print("\n[NORMAL TEST]")
    print(f"Current Temp: 92.0°C")
    print(f"Predicted Temp (90s): {predicted_temp_n:.2f}°C")
    print(f"Risk Score: {risk_n:.2f} | Alert: {fault_n}")
    
    # --- Test 2: Overheating Scenario ---
    predicted_temp_f, risk_f, fault_f, justification_f = hybrid_twin_prediction(test_data_fault, 'Overheating')
    print("\n[FAULT TEST (Overheating)]")
    print(f"Current Temp: 102.0°C")
    print(f"Predicted Temp (90s): {predicted_temp_f:.2f}°C")
    print(f"Risk Score: {risk_f:.2f} | Alert: {fault_f}")