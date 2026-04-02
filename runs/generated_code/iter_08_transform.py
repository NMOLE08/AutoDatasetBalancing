import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE

def main(input_path, output_path, target_column):
    try:
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

        # Use BorderlineSMOTE for oversampling
        borderline_smote = BorderlineSMOTE(random_state=42)
        X_resampled, y_resampled = borderline_smote.fit_resample(X_scaled, y)

        # Combine the resampled features and target into a DataFrame
        transformed_df = pd.DataFrame(X_resampled, columns=X.columns)
        transformed_df[target_column] = y_resampled

        # Save the transformed dataframe to output path
        transformed_df.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform dataset for better balance.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()
    main(args.input, args.output, args.target)
