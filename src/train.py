# Train script
import pandas as pd
import pickle
import argparse
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument("--model", required=True, help="Path to save trained model")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    with open(args.model, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {args.model}")
