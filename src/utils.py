import matplotlib.pyplot as plt
import pandas as pd

def plot_actual_vs_pred(actual: pd.Series, pred: pd.Series, title="Actual vs Predicted"):
    plt.figure(figsize=(8,4))
    plt.plot(actual.index, actual, label="Actual")
    plt.plot(actual.index, pred,   label="Predicted")
    plt.xlabel("Index")
    plt.ylabel("Price (€)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
