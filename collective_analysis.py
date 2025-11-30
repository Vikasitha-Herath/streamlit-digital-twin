import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "data"
CRITICAL_RISK_THRESHOLD = 0.7
FAULT_TEMP_THRESHOLD = 105.0

DATA_SCENARIOS = {
    "simulated_drive1.csv": "Normal (Stable)",
    "simulated_drive2.csv": "Fault (High Risk)",
    "simulated_drive3.csv": "Fault (Sensor Drift)",
    "simulated_drive4.csv": "Normal (Intermediate)",
    "simulated_drive5.csv": "Normal (Short)",
    "simulated_drive6.csv": "Fault (Med Risk)",
    "simulated_drive7.csv": "Normal (Long)",
}

def calculate_kpis(df, scenario_name):
    """Calculates key performance indicators for a single drive file, including Lead Time."""
    
    is_fault_scenario = "Fault" in scenario_name
    
    # 1. Alert Metrics
    max_risk = df['Prediction_Risk_Score'].max()
    avg_risk = df['Prediction_Risk_Score'].mean()
    critical_alerts = (df['Prediction_Risk_Score'] >= CRITICAL_RISK_THRESHOLD).sum()

    # 2. Lead Time Calculation (Prognostics Metric)
    lead_time_sec = None
    
    if is_fault_scenario:
        # a. Time of First Critical Alert
        critical_alert_rows = df[df['Prediction_Risk_Score'] >= CRITICAL_RISK_THRESHOLD]
        if not critical_alert_rows.empty:
            first_alert_time = critical_alert_rows['timestamp'].min()
        else:
            first_alert_time = None
            
        # b. Time of Physical Failure (Coolant Temp >= 105C)
        # We use the ACTUAL coolant temp here to check for system failure
        failure_rows = df[df['COOLANT_TEMPERATURE'] >= FAULT_TEMP_THRESHOLD]
        if not failure_rows.empty:
            first_failure_time = failure_rows['timestamp'].min()
        else:
            first_failure_time = None

        # c. Calculate Lead Time
        if first_alert_time is not None and first_failure_time is not None:
            if first_alert_time < first_failure_time:
                 lead_time_sec = first_failure_time - first_alert_time
            elif first_alert_time == first_failure_time:
                 lead_time_sec = 0.0
            else:
                 lead_time_sec = -1 # Alert came after physical failure (failure to warn)
        elif first_alert_time is None and first_failure_time is not None:
             lead_time_sec = -1 # Failure occurred, but no alert (system failure)
        elif first_alert_time is not None and first_failure_time is None:
             lead_time_sec = np.nan # Alert occurred, but physical failure never happened (false positive warning)

    return {
        'Scenario': scenario_name,
        'Max Risk': max_risk,
        'Avg Risk': avg_risk,
        'Critical Alerts (#)': critical_alerts,
        'Lead Time (s)': lead_time_sec
    }

def collective_analysis():
    """Main function to load all files and generate collective output."""
    all_results = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found. Please run 'python3 data_processor.py' first.")
        return
        
    for filename, scenario_name in DATA_SCENARIOS.items():
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping.")
            continue
            
        results = calculate_kpis(df, scenario_name)
        all_results.append(results)

    if not all_results:
        print("No files were successfully processed.")
        return

    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("      FINAL HYBRID DIGITAL TWIN COLLECTIVE PERFORMANCE SUMMARY")
    print("="*80)
    
    # Format the DataFrame for clean console output
    summary_df['Max Risk'] = summary_df['Max Risk'].map('{:.2f}'.format)
    summary_df['Avg Risk'] = summary_df['Avg Risk'].map('{:.2f}'.format)
    
    def format_lead_time(x):
        if pd.isna(x):
            return "N/A"
        elif x == -1:
            return "FAILED"
        else:
            return f"{x:.0f}"
            
    summary_df['Lead Time (s)'] = summary_df['Lead Time (s)'].apply(format_lead_time)
    
    print(summary_df.to_markdown(index=False))
    
    print("\n--- Summary Interpretation (The Academic Output) ---")
    print("Lead Time (s) > 0: Prognostics success (Alert occurred BEFORE physical failure).")
    print("Lead Time (s) = FAILED: Alert occurred AFTER physical failure, or not at all (System failure).")
    print("Lead Time (s) = N/A: Physical failure did not occur in this scenario.")
    print("Critical Alerts (#): Non-zero counts for Normal scenarios indicate potential False Positives.")
    print("="*80)

if __name__ == '__main__':
    collective_analysis()