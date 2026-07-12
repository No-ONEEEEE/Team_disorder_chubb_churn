# File: app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from eai import ChurnExplainabilityEngine 

# --- FLASK APP AND ENGINE INITIALIZATION ---
app = Flask(__name__)
# Enable CORS for frontend communication
CORS(app) 

engine = None
all_customer_data = None

try:
    engine = ChurnExplainabilityEngine(model_path='churn_model.pkl')
    all_customer_data = pd.read_csv('processed_features.csv')
    
    engine.initialize_explainer(all_customer_data)
    print("--- 🚀 Engine is ready and running! ---")
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}. Could not start API server.")
    engine = None


# --- MODIFIED /analyze ENDPOINT to handle global policies ---
@app.route('/analyze', methods=['POST']) # Changed from GET to POST
def analyze_churn():
    if engine is None:
        return jsonify({"error": "Engine is not initialized."}), 500
    
    # Get the policy from the request body. Default to None if not provided.
    policy = request.json.get('policy', None)
    
    # Work on a copy of the data to avoid modifying the original dataframe
    data_to_process = all_customer_data.copy()
    
    if policy and policy.get('feature') == 'CURR_ANN_AMT':
        try:
            percentage_change = float(policy.get('percentage', 0))
            multiplier = 1.0 + (percentage_change / 100.0)
            
            print(f"--- Applying Global Policy: Modifying CURR_ANN_AMT by {percentage_change}% ---")
            
            # 1. Apply the change to the primary feature
            data_to_process['CURR_ANN_AMT'] = data_to_process['CURR_ANN_AMT'] * multiplier
            
            # 2. CRITICAL: Recalculate any features that depend on the changed value
            # This ensures the simulation is accurate.
            if 'PREMIUM_TO_INCOME_RATIO' in data_to_process.columns:
                data_to_process['PREMIUM_TO_INCOME_RATIO'] = data_to_process['CURR_ANN_AMT'] / (data_to_process['INCOME'] + 1)
            
            if 'AMT_PER_DAY_TENURE' in data_to_process.columns:
                data_to_process['AMT_PER_DAY_TENURE'] = data_to_process['CURR_ANN_AMT'] / (data_to_process['DAYS_TENURE'] + 1)

        except Exception as e:
            print(f"❌ Error applying policy: {e}")
            return jsonify({"error": f"Failed to apply policy: {e}"}), 400

    # Process a sample for the main dashboard table (first 2000 rows)
    sample_data = data_to_process.head(2000)
    explanations = engine.batch_explain(sample_data)
    return jsonify(explanations)


# --- INDIVIDUAL SIMULATOR ENDPOINT (Unchanged) ---
@app.route('/simulate', methods=['POST'])
def simulate_churn():
    if engine is None:
        return jsonify({"error": "Engine is not initialized."}), 500

    try:
        modified_customer_data = request.json
        customer_template = all_customer_data.iloc[[0]].copy()

        for feature, value in modified_customer_data.items():
            val_to_use = 0.0 if value is None else float(value)
            if feature in customer_template.columns:
                 customer_template[feature] = val_to_use

        for col in customer_template.columns:
            customer_template[col] = pd.to_numeric(customer_template[col], errors='coerce').astype(float)
            
        explanation = engine.explain_prediction(customer_template)
        return jsonify(explanation)
    
    except Exception as e:
        print(f"Error in simulate_churn: {e}")
        return jsonify({"error": f"Simulation failed due to a backend error: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

