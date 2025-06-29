from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble    import RandomForestRegressor
from sklearn.metrics     import mean_absolute_error, mean_squared_error
from src.data_ingestion  import load_raw
from src.preprocessing   import clean_data
from src.features        import build_features
from src.utils           import plot_actual_vs_pred

def main():
    root       = Path(__file__).parent.parent
    data_path  = root/"/data"/"laptop_pricing_dataset_base.csv"
    model_path = root/"/models"/"price_model.pkl"

    # Load & clean
    df_raw   = load_raw(str(data_path))
    df_clean = clean_data(df_raw)

    # Features & target
    df_feat = build_features(df_clean)
    X = df_feat.drop("price_eur", axis=1)
    y = df_feat["price_eur"]

    # Split
    split = int(len(df_feat)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Eval
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Plot
    plot_actual_vs_pred(pd.Series(y_test.values, index=y_test.index),
                       pd.Series(preds,   index=y_test.index))

    # Save
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

if __name__=="__main__":
    main()
