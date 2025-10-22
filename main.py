# 🧶 YARN SUPPLY CHAIN DELAY PREDICTION + COST ANALYSIS

# NOTE: Install dependencies from your terminal, e.g.:
# pip install pandas numpy scikit-learn matplotlib seaborn lightgbm

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, roc_curve

parser = argparse.ArgumentParser(description="Run yarn supply chain analysis")
parser.add_argument("data_path", nargs="?", help="Path to yarn_supplychain_surat.csv")
args = parser.parse_args()

if not args.data_path:
    print("Usage: python main.py path/to/yarn_supplychain_surat.csv")
    sys.exit(1)

data_path = args.data_path

# Load dataset
df = pd.read_csv(
    data_path,
    parse_dates=["booking_date","scheduled_departure","scheduled_arrival","actual_departure","actual_arrival"]
)
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# Feature engineering
# sort by carrier and scheduled_departure so group rolling only looks within each carrier
df = df.sort_values(["carrier", "scheduled_departure"]).reset_index(drop=True)
df["lead_time_days"] = (df["scheduled_departure"] - df["booking_date"]).dt.days
df["planned_transit_days"] = (df["scheduled_arrival"] - df["scheduled_departure"]).dt.days
df["weekday_dep"] = df["scheduled_departure"].dt.weekday
df["month_dep"] = df["scheduled_departure"].dt.month

# rolling mean per carrier (ensure ordering by date within carrier)
df["carrier_delay_30"] = (
    df.groupby("carrier")["delay_flag"]
      .transform(lambda x: x.rolling(window=60, min_periods=1).mean())
      .fillna(0)
)

df["temp_sens_flag"] = df["temperature_sensitive"].map({"Yes": 1, "No": 0}).fillna(0)

features = [
    "planned_transit_days","lead_time_days","weekday_dep","month_dep",
    "carrier_delay_30","quantity_tonnes","shipping_cost",
    "expedite_surcharge","stockout_cost_per_tonne","temp_sens_flag"
]
X = df[features].fillna(-1)
y_clf = df["delay_flag"]
y_reg = df["delay_days"]

# 6️ Train/test split (chronological)
split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

# 7️ Train models
clf = RandomForestClassifier(n_estimators=120, random_state=42)
clf.fit(X_train, y_train_clf)
probs = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test_clf, probs)

reg = RandomForestRegressor(n_estimators=120, random_state=42)
reg.fit(X_train, y_train_reg)
pred = reg.predict(X_test)
mae = mean_absolute_error(y_test_reg, pred)

print("\n================ MODEL PERFORMANCE ================")
print(f" Delay Classifier ROC-AUC: {auc:.3f}")
print(f" Delay Regressor MAE: {mae:.2f} days")
print("===================================================")

# 8️ ROC Curve
fpr, tpr, _ = roc_curve(y_test_clf, probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", color='blue')
plt.plot([0,1],[0,1],'--', color='gray')
plt.title("ROC Curve for Delay Classification")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 9️ Delay Distribution
plt.figure(figsize=(7,4))
sns.histplot(df["delay_days"], kde=True, bins=20, color='steelblue')
plt.title("Distribution of Shipment Delay (Days)")
plt.xlabel("Delay Days")
plt.ylabel("Frequency")
plt.show()

#  Carrier performance
carrier_perf = df.groupby("carrier")["delay_days"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=carrier_perf.index, y=carrier_perf.values, palette="mako")
plt.title("Average Delay by Carrier")
plt.ylabel("Avg Delay (Days)")
plt.xticks(rotation=30)
plt.show()

# 11 Delay vs Shipping Cost
plt.figure(figsize=(7,5))
sns.scatterplot(x=df["shipping_cost"], y=df["delay_days"], hue=df["shipment_type"], alpha=0.7)
plt.title("Shipping Cost vs Delay Days")
plt.xlabel("Shipping Cost (₹)")
plt.ylabel("Delay Days")
plt.legend(title="Shipment Type")
plt.show()

# 12 Predict future shipments & evaluate expedite decisions
upcoming = df.tail(300).copy()
X_up = upcoming[features].fillna(-1)
upcoming["pred_delay_prob"] = clf.predict_proba(X_up)[:,1]
upcoming["pred_delay_days"] = reg.predict(X_up)
upcoming["expected_stockout_cost"] = (
    upcoming["pred_delay_prob"] * upcoming["quantity_tonnes"] * upcoming["stockout_cost_per_tonne"]
)
upcoming["expedite_cost"] = upcoming["shipping_cost"] + upcoming["expedite_surcharge"]
upcoming["baseline_expected_cost"] = upcoming["shipping_cost"] + upcoming["expected_stockout_cost"]
upcoming["expedite_decision"] = (upcoming["expedite_cost"] < upcoming["baseline_expected_cost"]).astype(int)

# 13 Cost comparison
total_baseline = upcoming["baseline_expected_cost"].sum()
total_after = (
    upcoming["expedite_decision"] * upcoming["expedite_cost"]
    + (1 - upcoming["expedite_decision"]) * upcoming["baseline_expected_cost"]
).sum()
savings = total_baseline - total_after

print("\n================ COST OPTIMIZATION ================")
print(f" Baseline Expected Cost: ₹{total_baseline:,.2f}")
print(f" Optimized Cost (After Decision): ₹{total_after:,.2f}")
print(f" Expected Savings: ₹{savings:,.2f}")
print(f" Expedite Shipments: {upcoming['expedite_decision'].sum()} out of {len(upcoming)}")
print("===================================================")

# 14 Cost Visualization
cost_df = pd.DataFrame({
    "Scenario": ["Baseline Cost", "Optimized Cost"],
    "Total_Cost": [total_baseline, total_after]
})
plt.figure(figsize=(6,4))
sns.barplot(x="Scenario", y="Total_Cost", data=cost_df, palette="crest")
plt.title("Cost Comparison: Baseline vs Optimized")
plt.ylabel("Total Cost (₹)")
plt.show()

# 15 Expedite decision impact
plt.figure(figsize=(6,4))
sns.countplot(x="expedite_decision", data=upcoming, palette="Set2")
plt.title("Expedite Decisions (0=No, 1=Yes)")
plt.xlabel("Decision")
plt.ylabel("Count of Shipments")
plt.show()

# 16 Delay probability distribution
plt.figure(figsize=(7,4))
sns.histplot(upcoming["pred_delay_prob"], bins=20, kde=True, color="orange")
plt.title("Predicted Delay Probability Distribution")
plt.xlabel("Predicted Probability of Delay")
plt.show()

