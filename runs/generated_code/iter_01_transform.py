import pandas as pd
import argparse
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def main(input_path, output_path, target_column):
    # Read the input CSV
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Impute missing values if any (though none are present in this dataset)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Use SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
    
    # Combine the resampled features and target into a DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled
    
    # Save the transformed dataframe to output CSV
    df_resampled.to_csv(output_path, index=False)

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
