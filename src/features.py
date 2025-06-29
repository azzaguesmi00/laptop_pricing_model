import pandas as pd

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature: price per GB of RAM.
    """
    df["price_per_ram_gb"] = df["price_eur"] / df["ram_gb"]
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.
    """
    df2 = add_ratio_features(df)
    # add more here as needed...
    return df2.dropna()
