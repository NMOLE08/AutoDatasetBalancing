import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import KMeansSMOTE
from sklearn.preprocessing import StandardScaler
import argparse

def main(input_path, output_path, target_column):
    # Read the input CSV
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Impute missing values (if any) using mean imputation
    X.fillna(X.mean(), inplace=True)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeansSMOTE for oversampling
    kmeans_smote = KMeansSMOTE(random_state=42)
    X_resampled, y_resampled = kmeans_smote.fit_resample(X_scaled, y)
    
    # Create a new DataFrame with resampled data
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled
    
    # Save the transformed dataframe to output path
    resampled_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and balance dataset.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output, args.target)
    except Exception as e:
        print(f"Transformation failed: {e}")
        exit(1)
