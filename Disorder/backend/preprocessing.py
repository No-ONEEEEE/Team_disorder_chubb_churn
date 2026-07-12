# File: preprocessing.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

def preprocess_data():
    """
    Loads raw data, cleans it, engineers features, and saves the final
    processed dataframes ready for model training.
    """
    print("--- Preprocessing Pipeline Started ---")

    # 1. Load All Data Files
    try:
        df_customer = pd.read_csv('archive/customer.csv')
        df_termination = pd.read_csv('archive/termination.csv')
        df_demographic = pd.read_csv('archive/demographic.csv')
        df_address = pd.read_csv('archive/address.csv')
        print("✅ 1/5: All data files loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please ensure all CSV files are in the current directory.")
        return

    # 2. Merge DataFrames and Create Target Variable
    df_termination['churn'] = np.where(df_termination['ACCT_SUSPD_DATE'].notna(), 1, 0)
    df = pd.merge(df_customer, df_termination[['INDIVIDUAL_ID', 'churn']], on='INDIVIDUAL_ID', how='left')
    df = pd.merge(df, df_demographic, on='INDIVIDUAL_ID', how='left')
    df = pd.merge(df, df_address, on='ADDRESS_ID', how='left')
    df['churn'].fillna(0, inplace=True)
    df['churn'] = df['churn'].astype(int)
    print("✅ 2/5: Data merged and target variable 'churn' created.")

    # 3. Feature Engineering
    coords = df[['LATITUDE', 'LONGITUDE']].fillna(df[['LATITUDE', 'LONGITUDE']].median())
    kmeans = KMeans(n_clusters=15, random_state=42, n_init='auto')
    df['LOCATION_CLUSTER'] = kmeans.fit_predict(coords)
    
    features_to_keep = [
        'CURR_ANN_AMT', 'DAYS_TENURE', 'AGE_IN_YEARS', 'INCOME',
        'LENGTH_OF_RESIDENCE', 'MARITAL_STATUS',
        'COLLEGE_DEGREE',
        'LOCATION_CLUSTER', 'churn'
    ]
    df_model = df[features_to_keep].copy()
    
    df_model['AMT_PER_DAY_TENURE'] = df_model['CURR_ANN_AMT'] / (df_model['DAYS_TENURE'] + 1)
    df_model['PREMIUM_TO_INCOME_RATIO'] = df_model['CURR_ANN_AMT'] / (df_model['INCOME'] + 1)
    df_model.replace([np.inf, -np.inf], 0, inplace=True)
    print("✅ 3/5: Feature engineering complete.")

    # 4. Data Cleaning (Imputation and Encoding)
    numerical_cols = df_model.select_dtypes(include=np.number).columns.drop('churn')
    categorical_cols = df_model.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        df_model[col].fillna(df_model[col].median(), inplace=True)
    for col in categorical_cols:
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)
    
    df_processed = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    print("✅ 4/5: Data cleaning and encoding complete.")

    # 5. Save Processed Data
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']

    X.to_csv('processed_features.csv', index=False)
    y.to_csv('processed_target.csv', index=False)
    print("✅ 5/5: Processed features and target saved to CSV files.")
    print("\n--- Preprocessing Finished ---")

if __name__ == "__main__":
    preprocess_data()