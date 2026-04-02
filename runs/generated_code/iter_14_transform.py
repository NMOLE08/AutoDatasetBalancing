import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTEN

def main(input_path, output_path, target_column):
    try:
        # Read the input CSV
        df = pd.read_csv(input_path)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Apply SMOTEN for oversampling
        smoten = SMOTEN(random_state=42)
        X_resampled, y_resampled = smoten.fit_resample(X_scaled, y)

        # Reconstruct the dataframe
        feature_columns = X.columns
        df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        df_resampled[target_column] = y_resampled

        # Save the transformed dataframe to output path
        df_resampled.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform dataset for better health score.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    args = parser.parse_args()

    main(args.input, args.output, args.target)
