def clean_data(df):
    # drop rows missing core specs
    return df.dropna(subset=["price_eur", "ram_gb", "storage_gb"])
