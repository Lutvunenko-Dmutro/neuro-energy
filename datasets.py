from static.mappings import DATASET_PATHS
import pandas as pd

def load_dataset(key):
    if key not in DATASET_PATHS:
        raise ValueError(f"Невідомий датасет: {key}")
    df = pd.read_csv(DATASET_PATHS[key])
    y = df["load"].values
    X = df.drop(columns=["load"]).values
    cols = df.drop(columns=["load"]).columns.tolist()
    return X, y, cols