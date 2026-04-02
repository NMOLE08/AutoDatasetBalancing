import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import argparse

def main(input_path, output_path, target_column):
    try:
        # Load the dataset
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

        # Apply NeighbourhoodCleaningRule for undersampling
        ncr = NeighbourhoodCleaningRule(random_state=42)
        X_resampled, y_resampled = ncr.fit_resample(X_scaled, y)

        # Reconstruct the DataFrame
        feature_columns = X.columns
        df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        df_resampled[target_column] = y_resampled

        # Save the transformed dataset
        df_resampled.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and balance a dataset.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()
    main(args.input, args.output, args.target)
