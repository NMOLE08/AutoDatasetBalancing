import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTETomek

def main(input_path, output_path, target_column):
    # Read the input CSV
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Impute missing values (if any)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Apply SMOTETomek for balancing
    smote_tomek = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smote_tomek.fit_resample(X_scaled, y)
    
    # Convert back to DataFrame
    balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
    balanced_df[target_column] = y_balanced
    
    # Ensure no NaN values in the output
    if balanced_df.isnull().values.any():
        raise ValueError("Output contains NaN values")
    
    # Save the transformed dataframe to output path
    balanced_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Balance and transform dataset.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.target)
    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)
