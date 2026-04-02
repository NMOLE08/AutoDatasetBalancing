import pandas as pd
import argparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks

def main(input_path, output_path, target_column):
    try:
        # Read the input CSV
        df = pd.read_csv(input_path)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Impute missing values if any (though none are present in this dataset)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Apply TomekLinks for undersampling
        tl = TomekLinks(random_state=42)
        X_resampled, y_resampled = tl.fit_resample(X_scaled, y)

        # Create a new DataFrame with resampled data
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df[target_column] = y_resampled

        # Save the transformed dataframe to output path
        resampled_df.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and balance dataset.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    args = parser.parse_args()

    main(args.input, args.output, args.target)
