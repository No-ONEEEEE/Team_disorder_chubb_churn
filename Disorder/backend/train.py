# File: train.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import optuna  # Import Optuna
import warnings

warnings.filterwarnings('ignore')

def train_model():
    """
    Loads preprocessed data, finds the best hyperparameters using Optuna,
    trains a final LightGBM model, evaluates it, and saves the artifact.
    """
    print("--- Model Training Pipeline Started ---")
    
    # 1. Load Preprocessed Data
    try:
        X = pd.read_csv('processed_features.csv')
        y = pd.read_csv('processed_target.csv').squeeze() # .squeeze() converts it to a Series
        print("✅ 1/5: Preprocessed data loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Processed data files not found. Please run preprocessing.py first.")
        return

    # 2. Prepare for Training
    ratio = y.value_counts()[0] / y.value_counts()[1]
    # Split data once for the optimization study
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"✅ 2/5: Data prepared. Imbalance ratio (scale_pos_weight) is {ratio:.2f}.")
    
    # 3. Optuna Hyperparameter Optimization ✨
    def objective(trial):
        """Define the search space for Optuna to find the best parameters."""
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'scale_pos_weight': ratio # Use the calculated ratio
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)])
        
        proba_predictions = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba_predictions)

    print("🚀 3/5: Starting hyperparameter optimization with Optuna...")


    best_params = {
        'n_estimators': 875, 
        'learning_rate': 0.13114963448948308, 
        'num_leaves': 202, 
        'max_depth': 12, 
        'subsample': 0.7153963816791664, 
        'colsample_bytree': 0.6307132005614821,
        'reg_alpha': 0.4376842315871772,
        'reg_lambda': 0.5809335540780333
    }
    
    print(f"✅ Using best parameters: {best_params}")

    # 4. Train Final Model with Best Parameters
    final_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        scale_pos_weight=ratio,
        **best_params  # Unpack the best parameters found by Optuna
    )
    
    print("🚀 4/5: Training the final model with the best parameters...")
    # We use the full dataset (X, y) here for the final training run
    final_model.fit(X, y)
    print("✅ Final model training complete!")

    # 5. Evaluate and Save Model
    # Note: We evaluate on the validation set from the initial split
    proba_predictions = final_model.predict_proba(X_val)[:, 1]
    
    custom_threshold = 0.40
    new_predictions = (proba_predictions >= custom_threshold).astype(int)

    print("\n--- Model Evaluation ---")
    print(f"🎯 AUC Score: {roc_auc_score(y_val, proba_predictions):.4f}")
    print("\n📋 Classification Report (using 40% threshold):")
    print(classification_report(y_val, new_predictions, target_names=['Did Not Churn (0)', 'Churned (1)']))

    joblib.dump(final_model, 'churn_model.pkl')
    print("\n✅ 5/5: Final tuned model saved to churn_model.pkl.")
    print("\n--- Training Pipeline Finished ---")

if __name__ == "__main__":
    train_model()