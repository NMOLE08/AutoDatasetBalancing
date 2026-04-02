import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import argparse

def main(input_path, output_path, target_column):
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Impute missing values if any (though none are present in this case)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Use RandomOverSampler for balancing
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
    
    # Combine back to DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled
    
    # Save the transformed dataframe
    resampled_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and balance dataset.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.target)
    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)
