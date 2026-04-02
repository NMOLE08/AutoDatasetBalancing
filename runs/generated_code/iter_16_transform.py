import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler

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

        # Identify categorical and numerical columns for SMOTENC
        categorical_features = [col_idx for col_idx, dtype in enumerate(df.dtypes) if dtype == 'int64' and df.iloc[:, col_idx].nunique() < 10]
        
        # Apply SMOTENC to balance the dataset
        smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
        X_resampled, y_resampled = smote_nc.fit_resample(X_scaled, y)

        # Create a new DataFrame with resampled data
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled

        # Save the transformed dataframe to output path
        df_resampled.to_csv(output_path, index=False)

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
