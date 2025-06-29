import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing core specs/target and clip negatives.
    """
    df2 = df.dropna(subset=["price_eur","ram_gb","storage_gb"])
    df2["ram_gb"]     = df2["ram_gb"].clip(lower=0)
    df2["storage_gb"] = df2["storage_gb"].clip(lower=0)
    return df2

