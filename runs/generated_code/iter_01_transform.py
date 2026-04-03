import os

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import ADASYN

def enforce_domain_rules(df, domain_rules):
    for column, rules in domain_rules.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df[column].fillna((rules['min'] + rules['max']) // 2, inplace=True)
            df[column] = np.round(df[column]).astype(int).clip(lower=rules['min'], upper=rules['max'])
    return df

def main():
    parser = argparse.ArgumentParser(description='Transform dataset for balanced model training.')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--target', required=True, help='Target column name')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        
        # Replace string representations of missing data with np.nan
        df.replace('?', np.nan, inplace=True)
        
        # Convert numeric columns that were read as object/string back to numeric
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cast integer columns with NaNs to pandas nullable integer type
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype('Int64')
        
        # Impute all NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Separate features and target
        X = df_imputed.drop(columns=[args.target])
        y = df_imputed[args.target]
        
        # Apply ADASYN balancing technique
        ada = ADASYN(random_state=42)
        X_resampled, y_resampled = ada.fit_resample(X, y)
        
        # Combine resampled features and target
        df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=args.target)], axis=1)
        
        # Enforce domain rules
        domain_rules_json = json.loads(os.getenv('DOMAIN_RULES_JSON', '{}'))
        df_balanced = enforce_domain_rules(df_balanced, domain_rules_json)
        
        # Verify output contains no NaN in feature columns
        if df_balanced.drop(columns=[args.target]).isnull().values.any():
            raise ValueError("Unresolved NaN values found in feature columns after balancing.")
        
        # Save transformed CSV to --output
        df_balanced.to_csv(args.output, index=False)
    
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()
