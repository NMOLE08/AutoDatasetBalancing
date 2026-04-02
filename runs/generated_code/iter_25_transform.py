import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sys
import argparse

def main(input_path, output_path, target_column):
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Impute missing values
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    imputer = SimpleImputer(strategy='median')
    X[numeric_features] = imputer.fit_transform(X[numeric_features])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use BalancedRandomForestClassifier with class_weight='balanced'
    brf = BalancedRandomForestClassifier(class_weight='balanced', random_state=42)
    brf.fit(X_scaled, y)
    
    # Transform the dataset using SMOTE and RandomUnderSampler
    smote = SMOTE(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    
    # Create a DataFrame from the resampled data
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced = pd.concat([X_resampled_df, pd.Series(y_resampled, name=target_column)], axis=1)
    
    # Save the transformed dataframe to output path
    df_balanced.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and balance dataset.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.target)
    except Exception as e:
        print(f"Transformation failed: {e}", file=sys.stderr)
        sys.exit(1)
