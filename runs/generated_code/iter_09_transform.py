import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE
from argparse import ArgumentParser

def main(input_path, output_path, target_column):
    # Read the input CSV
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Impute missing values if any (though none are present in this dataset)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Use SVMSMOTE for oversampling
    svm_smote = SVMSMOTE(random_state=42)
    X_resampled, y_resampled = svm_smote.fit_resample(X_scaled, y)
    
    # Combine the resampled features and target into a DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled = pd.concat([X_resampled_df, y_resampled], axis=1)
    df_resampled.rename(columns={0: target_column}, inplace=True)
    
    # Save the transformed dataframe to output path
    df_resampled.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV input file path')
    parser.add_argument('--output', required=True, help='CSV output file path')
    parser.add_argument('--target', required=True, help='Target column name')
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.target)
    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)
