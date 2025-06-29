import streamlit as st
import joblib
import pandas as pd
from src.data_ingestion import load_raw
from src.preprocessing  import clean_data
from src.features       import build_features

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("/models/price_model.pkl")

def main():
    st.title("Laptop Price & UP/LOSS Flagger")

    df = load_raw("/data/laptop_pricing_dataset_base.csv")
    df = clean_data(df)
    df_feat = build_features(df)

    model = load_model()
    st.write("Dataset samples:")
    st.dataframe(df_feat.head())

    if st.button("Show Predictions"):
        X = df_feat.drop("price_eur", axis=1)
        y = df_feat["price_eur"]
        preds = model.predict(X)
        df_out = df_feat.copy()
        df_out["predicted_price"] = preds
        df_out["flag"] = ["LOSS" if p<y0 else "UP" 
                          for p,y0 in zip(preds, y)]
        st.dataframe(df_out.head(20))

if __name__=="__main__":
    main()
