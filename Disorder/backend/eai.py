# File: explainable_churn_ai.py

import pandas as pd
import numpy as np
import joblib
import shap
import warnings
from typing import Dict, List, Any
import json
from datetime import datetime

warnings.filterwarnings('ignore')


class ChurnExplainabilityEngine:
    """
    Main engine for generating explainable AI insights for churn predictions.
    """
    
    def __init__(self, model_path: str = 'churn_model.pkl'):
        """Initialize the explainability engine."""
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.feature_descriptions = {}
        
        print("🚀 Initializing Explainable AI Engine...")
        self._load_model(model_path)
        print("✅ Explainability Engine Ready!")
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Model not found at {model_path}. Please train the model first."
            )
    
    def _generate_feature_descriptions(self, feature_names: List[str]) -> Dict[str, str]:
        """Dynamically creates human-readable descriptions from actual feature names."""
        descriptions = {}
        for name in feature_names:
            desc = name.replace('_', ' ').lower()
            
            if name == 'CURR_ANN_AMT':
                desc = 'current annual premium'
            elif name == 'DAYS_TENURE':
                desc = 'policy tenure in days'
            elif name == 'AGE_IN_YEARS':
                desc = 'customer age'
            elif 'RATIO' in name:
                desc = name.replace('_', ' ').lower()
            elif 'MARITAL_STATUS' in name:
                status = name.split('_')[-1]
                desc = f'marital status ({status})'
            elif 'HOME_OWNER' in name:
                status = name.split('_')[-1]
                desc = f'home ownership ({status})'
            elif 'LOCATION_CLUSTER' in name:
                desc = 'geographic location cluster'

            descriptions[name] = desc
        return descriptions

    def initialize_explainer(self, X_train: pd.DataFrame):
        """Initialize SHAP explainer."""
        print("🔧 Initializing SHAP explainer...")
        self.feature_names = X_train.columns.tolist()
        self.feature_descriptions = self._generate_feature_descriptions(self.feature_names)
        
        # This is a more robust way to initialize for LightGBM models
        self.explainer = shap.TreeExplainer(self.model)
        
        print("✅ SHAP explainer initialized successfully!")
    
    def explain_prediction(
        self, 
        customer_data: pd.DataFrame,
        threshold: float = 0.40
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation for a single customer."""
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call initialize_explainer() first.")
        
        churn_probability = self.model.predict_proba(customer_data)[0, 1]
        churn_prediction = int(churn_probability >= threshold)
        
        shap_values = self.explainer.shap_values(customer_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        explanation = {
            'customer_id': int(customer_data.index[0]), # Fix for JSON serialization
            'churn_probability': round(float(churn_probability), 4),
            'churn_prediction': churn_prediction,
            'risk_level': self._categorize_risk(churn_probability),
            'timestamp': datetime.now().isoformat(),
            'feature_contributions': self._get_feature_contributions(customer_data, shap_values[0]),
            'reasoning': self._generate_reasoning(customer_data, shap_values[0], churn_probability),
            'retention_recommendations': self._generate_recommendations(customer_data, shap_values[0]),
            'confidence_metrics': {
                'base_rate': round(float(base_value), 4),
                'confidence_level': self._assess_confidence(shap_values[0])
            }
        }
        return explanation

    def _get_feature_contributions(self, customer_data: pd.DataFrame, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Get sorted list of feature contributions."""
        contributions = []
        for idx, feature in enumerate(self.feature_names):
            contributions.append({
                'feature': feature,
                'feature_value': self._format_feature_value(customer_data[feature].iloc[0]),
                'shap_value': round(float(shap_values[idx]), 4),
                'impact': 'increases risk' if shap_values[idx] > 0 else 'decreases risk',
                'magnitude': abs(float(shap_values[idx]))
            })
        contributions.sort(key=lambda x: x['magnitude'], reverse=True)
        return contributions

    def _generate_reasoning(self, customer_data: pd.DataFrame, shap_values: np.ndarray, probability: float) -> str:
        """Generate human-readable reasoning for the prediction."""
        contributions = self._get_feature_contributions(customer_data, shap_values)
        top_factor = contributions[0]
        risk_level = self._categorize_risk(probability)
        
        narrative = f"This customer has a {risk_level} churn risk ({probability:.1%} probability). "
        narrative += f"The primary factor is their {self.feature_descriptions.get(top_factor['feature'], top_factor['feature'])} "
        narrative += f"of '{top_factor['feature_value']}', which significantly {top_factor['impact']}."
        return narrative
    
    # In ChurnExplainabilityEngine class in eai.py

    def _generate_recommendations(self, customer_data: pd.DataFrame, shap_values: np.ndarray) -> List[str]:
        """Generate actionable, context-aware retention recommendations."""
        recommendations = []
        contributions = self._get_feature_contributions(customer_data, shap_values)
        probability = self.model.predict_proba(customer_data)[0, 1]

        # Filter for factors that increase risk
        risk_factors = [c for c in contributions if c['impact'] == 'increases risk']
        
        if not risk_factors:
            return ["✅ Customer is low risk - maintain current engagement."]

        # --- NEW CONTEXT-AWARE RULES ---
        for factor in risk_factors[:3]: # Look at top 3 risk drivers
            feature = factor['feature']
            value_str = factor['feature_value']
            # Convert value back to a number for checks
            try:
                value = float(str(value_str).replace(',', ''))
            except ValueError:
                value = 0

            # Rule 1: High Premium
            if 'AMT' in feature.upper():
                recommendations.append("🚗 Review policy for potential safe driver or multi-car discounts.")

            # Rule 2: Affordability Issue
            if 'PREMIUM_TO_INCOME_RATIO' in feature.upper():
                recommendations.append("💰 Offer flexible payment plans or suggest a policy review for better value.")
            
            # Rule 3: New Customer Risk (Low Tenure)
            if 'TENURE' in feature.upper() and value < 365: # e.g., less than a year
                recommendations.append("👋 Proactively schedule a 'new customer check-in' call to build loyalty.")

            # Rule 4: Loyal Customer Risk (High Tenure but still a risk factor)
            if 'TENURE' in feature.upper() and value > 1825: # e.g., more than 5 years
                recommendations.append("🏆 Offer a long-term loyalty reward, like a complimentary service.")
            
            # Rule 5: Location-based Risk
            if 'LOCATION_CLUSTER' in feature.upper():
                recommendations.append("🗺️ Investigate regional competition and consider a location-specific promotion.")

        # --- NEW GENERAL RECOMMENDATIONS BASED ON RISK ---
        if probability >= 0.7:
            recommendations.append("🚨 URGENT: This is a critical risk customer. Personal outreach required within 24 hours.")
        elif probability >= 0.5:
            recommendations.append("📞 This is a high-risk customer. Add to this week's retention campaign call list.")
        
        # Return a unique list of recommendations
        if recommendations:
            return list(set(recommendations))
        else:
            return ["Monitor customer engagement for any changes."]

    def _format_feature_value(self, value: Any) -> str:
        if isinstance(value, (int, np.integer)): return str(value)
        elif isinstance(value, (float, np.floating)): return f"{value:,.2f}"
        return str(value)

    def _categorize_risk(self, probability: float) -> str:
        if probability >= 0.7: return "HIGH"
        elif probability >= 0.4: return "MODERATE"
        else: return "LOW"

    def _assess_confidence(self, shap_values: np.ndarray) -> str:
        total_impact = np.sum(np.abs(shap_values))
        top_3_impact = np.sum(np.sort(np.abs(shap_values))[-3:])
        concentration_ratio = top_3_impact / (total_impact + 1e-10)
        
        if concentration_ratio > 0.7: return "HIGH - Few dominant factors"
        elif concentration_ratio > 0.5: return "MODERATE - Several contributing factors"
        else: return "LOW - Many small factors"

    # --- Replace the entire old batch_explain method with this ---

    def batch_explain(self, customer_data: pd.DataFrame, save_path: str = None) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple customers using efficient batch processing.
        """
        print(f"🔍 Generating explanations for {len(customer_data)} customers...")
        
        # --- OPTIMIZATION ---
        # 1. Get all predictions and SHAP values in one batch operation (much faster)
        all_probabilities = self.model.predict_proba(customer_data)[:, 1]
        all_shap_values = self.explainer.shap_values(customer_data)
        if isinstance(all_shap_values, list):
            all_shap_values = all_shap_values[1] # Use positive class
        
        explanations = []
        # 2. Now loop through the *results* to build the JSON (this part is fast)
        for idx in range(len(customer_data)):
            customer_row = customer_data.iloc[[idx]]
            
            # We are now re-using the pre-calculated values, not re-calculating them
            churn_probability = all_probabilities[idx]
            shap_values = all_shap_values[idx:idx+1] # Keep shape for internal functions
            
            # --- This reuses your existing logic without re-calculating ---
            explanation = {
                'customer_id': int(customer_row.index[0]),
                'churn_probability': round(float(churn_probability), 4),
                'churn_prediction': int(churn_probability >= 0.40), # Using your threshold
                'risk_level': self._categorize_risk(churn_probability),
                'timestamp': datetime.now().isoformat(),
                'feature_contributions': self._get_feature_contributions(customer_row, shap_values[0]),
                'reasoning': self._generate_reasoning(customer_row, shap_values[0], churn_probability),
                'retention_recommendations': self._generate_recommendations(customer_row, shap_values[0]),
                'confidence_metrics': {
                    'base_rate': round(float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value), 4),
                    'confidence_level': self._assess_confidence(shap_values[0])
                }
            }
            explanations.append(explanation)
            
            if (idx + 1) % 500 == 0:
                print(f"  ...packaged {idx + 1}/{len(customer_data)} explanations")
        
        print(f"✅ Generated {len(explanations)} explanations.")
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(explanations, f, indent=2)
            print(f"💾 Explanations saved to {save_path}")
        
        return explanations
    
    def get_global_feature_importance(self) -> pd.DataFrame:
        """Calculate global feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance['importance_percentage'] = (
                importance['importance'] / importance['importance'].sum() * 100
            ).round(2)
            
            return importance
        return None

def main():
    """
    Main function to run the Explainable AI system on a SAMPLE of 2,000 customers.
    """
    print("=" * 70)
    print("   SAMPLE DATASET EXPLAINABILITY ANALYSIS (2,000 Customers)")
    print("=" * 70)
    
    engine = ChurnExplainabilityEngine(model_path='churn_model.pkl')
    
    try:
        # <-- THIS LINE IS CHANGED to load only 2,000 records -->
        all_customer_data = pd.read_csv('processed_features.csv').head(2000)
        print(f"✅ Loaded a sample of {len(all_customer_data)} customer records.")
    except FileNotFoundError:
        print("❌ `processed_features.csv` not found. Please run preprocessing.py first.")
        return

    engine.initialize_explainer(all_customer_data)

    print("\n" + "=" * 70)
    print(f"BATCH PROCESSING A SAMPLE OF {len(all_customer_data)} CUSTOMERS")
    print("=" * 70)
    
    # <-- THIS LINE IS CHANGED to save to a different file -->
    all_explanations = engine.batch_explain(
        all_customer_data,
        save_path='sample_2k_explanations.json'
    )

    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 70)

    importance = engine.get_global_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ SAMPLE ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n📊 A JSON with explanations for 2,000 customers has been saved to:")
    print("   --> sample_2k_explanations.json")

if __name__ == '__main__':
    main()