import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import BorderlineSMOTE

def parse_arguments():
    parser = argparse.ArgumentParser(description='Transform and balance dataset.')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    try:
        # Read input data
        df = pd.read_csv(args.input)
        
        # Separate features and target
        X = df.drop(columns=[args.target])
        y = df[args.target]
        
        # Impute missing values (if any) using mean imputation
        X_imputed = X.fillna(X.mean())
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Apply BorderlineSMOTE for oversampling
        smote = BorderlineSMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        
        # Convert back to DataFrame
        X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced = pd.concat([X_resampled_df, y_resampled], axis=1)
        
        # Ensure no NaN values in the output
        if df_balanced.isnull().values.any():
            raise ValueError("Output contains NaN values")
        
        # Save the transformed dataframe to output CSV
        df_balanced.to_csv(args.output, index=False)
    
    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
