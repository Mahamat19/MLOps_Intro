import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw JSON")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV")
    args = parser.parse_args()

    # Read JSON instead of CSV
    df = pd.read_json(args.input)

    # Example preprocessing: multiply first numeric column by 2
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        df[numeric_cols[0]] = df[numeric_cols[0]] * 2

    df.to_csv(args.output, index=False)
    print(f"Preprocessed data saved to {args.output}")
