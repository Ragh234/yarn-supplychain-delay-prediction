🧶 Yarn Supply Chain Delay Prediction & Cost Optimization
📘 Overview

This project focuses on predicting shipment delays and minimizing overall logistics costs in the Yarn Manufacturing and Distribution Supply Chain.
The yarn company, based in Surat, processes raw yarn sourced from Indian states like Gujarat, Chennai, Punjab, Rajasthan, and UP, and distributes finished yarn to Mumbai, Madhya Pradesh, Assam, Gujarat, Kerala, West Bengal, Karnataka, Jaipur, and Uttarakhand.

🎯 Objectives

Predict the probability and duration of shipment delays using historical data.

Estimate cost impacts due to delays, stockouts, and expedited shipping.

Recommend optimal expedite decisions to minimize total logistics cost.

Visualize insights via graphs and Power BI dashboards.

🧩 Project Workflow

Data Preparation

Synthetic dataset (yarn_supplychain_surat.csv) with ~1000 records.

Includes features such as:

booking_date, scheduled_departure, actual_arrival

carrier, shipment_type, weight, quantity_tonnes

shipping_cost, expedite_surcharge, stockout_cost_per_tonne

delay_flag, delay_days

Data simulated to represent Surat-based factory operations.

Feature Engineering

Calculated lead time, planned transit days, month, and weekday.

Added rolling carrier delay rate (30-day window).

Encoded categorical fields like temperature sensitivity.

Modeling

RandomForestClassifier → predicts probability of shipment delay.

RandomForestRegressor → predicts expected delay days.

Model evaluation metrics:

ROC-AUC for classification

MAE (Mean Absolute Error) for regression

Optimization

Calculated baseline vs optimized costs using predicted delays:

Expected Stockout Cost = Delay Probability × Quantity × Stockout Cost
Expedite Cost = Shipping Cost + Expedite Surcharge
Decision: Expedite if Expedite Cost < Expected Stockout Cost


Computed total baseline cost, optimized cost, and savings.

📊 Visualizations (Matplotlib + Seaborn)

The notebook produces multiple insightful graphs:

Visualization	Purpose
ROC Curve	Classifier performance (AUC score)
Delay Distribution	Frequency of shipment delays
Average Delay by Carrier	Carrier efficiency comparison
Shipping Cost vs Delay Days	Correlation insight
Cost Comparison (Baseline vs Optimized)	Savings visualization
Expedite Decision Counts	Number of expedited vs normal shipments
Predicted Delay Probability	Future risk assessment
🧠 Key Outputs

Delay Classifier ROC-AUC: ~0.82

Delay Regressor MAE: ~1.4 days

Expected Cost Savings: ₹200K–₹500K (depending on data)

Expedite Recommendations: Clear, data-driven decision support

🧰 Tech Stack

Language: Python 3.10

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, lightgbm

Platform: Google Colab

Visualization Tools: Matplotlib, Seaborn, Power BI

🚀 How to Run

Open Google Colab
.

Upload the provided notebook and dataset (yarn_supplychain_surat.csv).

Run all cells in order.

Observe printed outputs and visual graphs directly.
