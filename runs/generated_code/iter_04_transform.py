import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import argparse

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
    
    # Apply BorderlineSMOTE for balancing
    smote = BorderlineSMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    # Combine the balanced features and target back into a DataFrame
    balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
    balanced_df[target_column] = y_balanced
    
    # Save the transformed dataframe to output CSV
    balanced_df.to_csv(output_path, index=False)

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
