import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import ClusterCentroids

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

        # Use ClusterCentroids for undersampling
        cc = ClusterCentroids(random_state=42)
        X_resampled, y_resampled = cc.fit_resample(X_imputed, y)

        # Combine resampled features and target back into a DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled

        # Save the transformed dataframe to output path
        df_resampled.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform dataset for better balance.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()
    main(args.input, args.output, args.target)
