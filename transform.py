from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in input dataset.")

    # Default fallback: lightweight cleanup only.
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    for col in df.columns:
        if col == args.target:
            continue
        if df[col].dtype.kind in {"i", "u", "f"}:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

