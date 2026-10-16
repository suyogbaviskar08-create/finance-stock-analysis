"""Run the full pipeline end to end: python run_analysis.py

Outputs: figures in reports/figures/ and metrics in reports/metrics.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import load_apple, load_stocks
from src.features import build_features
from src.models import (chronological_split, classification_report_table, fit_classifiers,
                        fit_regressors, performance, regression_report)

OUT = Path(__file__).resolve().parent / "reports"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def eda_figures(df, px):
    # 1. Price, moving averages, volume
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(df.index, df["Close"], label="Close", lw=1.4)
    for n in (20, 50, 200):
        a1.plot(df.index, df["Close"].rolling(n).mean(), label=f"{n}-day MA", lw=1)
    a1.set_title("AAPL closing price with moving averages"); a1.legend(ncol=4)
    a2.bar(df.index, df["Volume"] / 1e6, color="#8a9bb8", width=1.0); a2.set_ylabel("Volume (M)")
    save("01_price_volume.png")

    # 2. Return distribution
    r = df["Return"].dropna() * 100
    plt.figure(figsize=(8, 4.5)); sns.histplot(r, bins=60, kde=True)
    plt.title("Distribution of daily returns (%)"); save("02_return_distribution.png")

    # 3. Rolling volatility
    vol = df["Return"].rolling(20).std() * np.sqrt(252) * 100
    plt.figure(figsize=(11, 4)); plt.plot(vol, color="#c0392b")
    plt.title("Rolling 20-day annualised volatility (%)"); save("03_rolling_volatility.png")

    # 4. Multi-stock comparison
    common = px.dropna()
    rebased = common / common.iloc[0] * 100
    dd = (px / px.cummax() - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    rebased.plot(ax=axes[0], title="Growth of 100 invested")
    dd.plot(ax=axes[1], title="Drawdown from previous peak (%)")
    save("04_stock_comparison.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(px.pct_change().corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.2, vmax=1)
    plt.title("Correlation of monthly returns"); save("05_correlation_heatmap.png")


def main():
    df, px = load_apple(), load_stocks()
    eda_figures(df, px)

    data, features = build_features(df)
    train, test = chronological_split(data)
    X_tr, X_te = train[features], test[features]
    print(f"Train {train.index.min().date()}..{train.index.max().date()} ({len(train)}) | "
          f"Test {test.index.min().date()}..{test.index.max().date()} ({len(test)})")

    # ---- Regression: next-day return
    regs = fit_regressors(X_tr, train["target_ret"])
    reg_preds = {"Naive (0 return)": np.zeros(len(test))}
    reg_preds.update({k: m.predict(X_te) for k, m in regs.items()})
    close_today = df["Close"].reindex(test.index).values
    reg_table = regression_report(test["target_ret"], reg_preds, close_today)
    print("\nRegression (next-day return)\n", reg_table.round(3))

    # ---- Classification: next-day direction
    clfs = fit_classifiers(X_tr, train["target_up"])
    majority = int(train["target_up"].mean() >= 0.5)
    clf_preds = {"Baseline (majority class)": np.full(len(test), majority)}
    clf_preds.update({k: m.predict(X_te) for k, m in clfs.items()})
    clf_table = classification_report_table(test["target_up"], clf_preds)
    print("\nClassification (next-day direction)\n", clf_table.round(2))
    print(f"Always-'up' guess would score {test['target_up'].mean() * 100:.1f}%")

    # ---- Back-test of best classifier
    best = clf_table.drop("Baseline (majority class)")["Test accuracy (%)"].idxmax()
    sig = pd.Series(clf_preds[best], index=test.index)
    strat, hold = sig * test["target_ret"], test["target_ret"]
    bt = pd.DataFrame({"Model strategy": performance(strat), "Buy & hold": performance(hold)})
    print(f"\nBack-test using {best}\n", bt.round(2))

    # ---- Figures from modelling
    best_reg = reg_table.drop("Naive (0 return)")["Return RMSE (%)"].idxmin()
    plt.figure(figsize=(11, 4.5))
    plt.plot(test.index, close_today * (1 + test["target_ret"].values), label="Actual close")
    plt.plot(test.index, close_today * (1 + reg_preds[best_reg]), label=f"{best_reg} forecast")
    plt.title("Next-day close: actual vs forecast (test period)"); plt.legend()
    save("06_forecast_vs_actual.png")

    plt.figure(figsize=(8, 4))
    imp = pd.Series(clfs["Random Forest"].best_estimator_.feature_importances_, index=features).sort_values()
    imp.plot(kind="barh", title="Random Forest feature importance (direction)")
    save("07_feature_importance.png")

    plt.figure(figsize=(11, 4))
    ((1 + strat).cumprod() * 100).plot(label="Model strategy")
    ((1 + hold).cumprod() * 100).plot(label="Buy & hold")
    plt.title("Back-test, test period (start = 100)"); plt.legend()
    save("08_backtest.png")

    metrics = {
        "regression": reg_table.round(4).to_dict("index"),
        "classification": clf_table.round(3).to_dict("index"),
        "backtest": bt.round(3).to_dict(),
        "always_up_accuracy_pct": round(float(test["target_up"].mean() * 100), 2),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved figures to {FIG} and metrics to {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
