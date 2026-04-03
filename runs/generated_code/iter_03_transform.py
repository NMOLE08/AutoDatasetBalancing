import pandas as pd
import argparse
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import BorderlineSMOTE

def main():
    parser = argparse.ArgumentParser(description='Transform and balance dataset.')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--target', required=True, help='Target column name')
    args = parser.parse_args()

    try:
        # Read the input CSV
        df = pd.read_csv(args.input)

        # Separate features and target
        X = df.drop(columns=[args.target])
        y = df[args.target]

        # Impute missing values if any (though none are present in this dataset)
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Apply BorderlineSMOTE for oversampling
        smote = BorderlineSMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

        # Combine resampled features and target into a DataFrame
        transformed_df = pd.DataFrame(X_resampled, columns=X.columns)
        transformed_df[args.target] = y_resampled

        # Save the transformed dataframe to output CSV
        transformed_df.to_csv(args.output, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == '__main__':
    main()
