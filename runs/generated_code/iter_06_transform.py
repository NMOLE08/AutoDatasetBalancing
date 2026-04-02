import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from argparse import ArgumentParser

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

        # Use BorderlineSMOTE for oversampling
        smote = BorderlineSMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Combine the resampled features and target into a DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled

        # Save the transformed dataframe to output CSV
        df_resampled.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()

    main(args.input, args.output, args.target)
