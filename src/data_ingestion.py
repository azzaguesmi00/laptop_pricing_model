# src/data_ingestion.py
import pandas as pd

def load_raw(path="/data/laptop_pricing_dataset_base.csv") -> pd.DataFrame:
    """
    Load CSV without header, assign proper column names, coerce to numeric.
    """
    cols = [
        "brand","type_code","screen_type","code_3","code_4","code_5",
        "screen_cm","weight_kg","ram_gb","storage_gb","code_10","price_eur"
    ]
    df = pd.read_csv(path, header=None, names=cols)
    for c in ["screen_cm","weight_kg","ram_gb","storage_gb","price_eur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



