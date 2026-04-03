import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from argparse import ArgumentParser

def enforce_domain_rules(df, domain_rules):
    for column, rules in domain_rules.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df[column].fillna((rules['min'] + rules['max']) // 2, inplace=True)
            df[column] = np.round(df[column]).astype(int).clip(lower=rules['min'], upper=rules['max'])
    return df

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', required=True)
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
            if df[col].dropna().apply(lambda x: x.is_integer()).all():
                df[col] = df[col].astype('Int64')

        # Impute all NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Apply SMOTE for balancing
        X = df_imputed.drop(columns=[args.target])
        y = df_imputed[args.target]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Enforce domain rules
        domain_rules_json = '{"erythema": {"type": "int", "min": 0, "max": 3}, "scaling": {"type": "int", "min": 0, "max": 3}, "definite_borders": {"type": "int", "min": 0, "max": 3}, "itching": {"type": "int", "min": 0, "max": 3}, "koebner_phenomenon": {"type": "int", "min": 0, "max": 3}, "polygonal_papules": {"type": "int", "min": 0, "max": 3}, "follicular_papules": {"type": "int", "min": 0, "max": 3}, "oral_mucosal_involvement": {"type": "int", "min": 0, "max": 3}, "knee_and_elbow_involvement": {"type": "int", "min": 0, "max": 3}, "scalp_involvement": {"type": "int", "min": 0, "max": 3}, "family_history": {"type": "int", "min": 0, "max": 3}, "melanin_incontinence": {"type": "int", "min": 0, "max": 3}, "eosinophils_infiltrate": {"type": "int", "min": 0, "max": 3}, "PNL_infiltrate": {"type": "int", "min": 0, "max": 3}, "fibrosis_papillary_dermis": {"type": "int", "min": 0, "max": 3}, "exocytosis": {"type": "int", "min": 0, "max": 3}, "acanthosis": {"type": "int", "min": 0, "max": 3}, "hyperkeratosis": {"type": "int", "min": 0, "max": 3}, "parakeratosis": {"type": "int", "min": 0, "max": 3}, "clubbing_rete_ridges": {"type": "int", "min": 0, "max": 3}, "elongation_rete_ridges": {"type": "int", "min": 0, "max": 3}, "thinning_suprapapillary_epidermis": {"type": "int", "min": 0, "max": 3}, "spongiform_pustule": {"type": "int", "min": 0, "max": 3}, "munro_microabcess": {"type": "int", "min": 0, "max": 3}, "focal_hypergranulosis": {"type": "int", "min": 0, "max": 3}, "disappearance_granular_layer": {"type": "int", "min": 0, "max": 3}, "vacuolisation_damage_basal_layer": {"type": "int", "min": 0, "max": 3}, "spongiosis": {"type": "int", "min": 0, "max": 3}, "saw_tooth_appearance_retes": {"type": "int", "min": 0, "max": 3}, "follicular_horn_plug": {"type": "int", "min": 0, "max": 3}, "perifollicular_parakeratosis": {"type": "int", "min": 0, "max": 3}, "inflammatory_mononuclear_infiltrate": {"type": "int", "min": 0, "max": 3}, "band_like_infiltrate": {"type": "int", "min": 0, "max": 3}, "age": {"type": "int", "min": 0, "max": 120}, "class": {"type": "int", "min": 1, "max": 6}}'
        domain_rules = eval(domain_rules_json)
        X_resampled = enforce_domain_rules(X_resampled, domain_rules)

        # Combine resampled data
        df_balanced = pd.concat([X_resampled, y_resampled], axis=1)

        # Save transformed CSV to --output
        df_balanced.to_csv(args.output, index=False)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()