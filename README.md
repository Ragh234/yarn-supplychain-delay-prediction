# Yarn Supply Chain Delay Prediction & Cost Optimization

[![CI](https://github.com/Ragh234/yarn-supplychain-delay-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Ragh234/yarn-supplychain-delay-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts shipment delays for a yarn manufacturer's inbound/outbound
logistics (modeled on a Surat-based operation sourcing from Gujarat,
Chennai, Punjab, Rajasthan, and UP, and distributing to Mumbai, MP, Assam,
Kerala, West Bengal, Karnataka, and Uttarakhand), then uses those
predictions to recommend which shipments are worth expediting.

**The dataset (`yarn_supplychain_surat.csv`, ~1,000 rows) is synthetic** —
generated to be structurally realistic, not scraped or sourced from a real
carrier. Treat all figures below as a demonstration of the modeling and
cost-optimization approach, not a claim about real-world logistics
accuracy.

## Run it

```bash
pip install -r requirements.txt
python main.py yarn_supplychain_surat.csv
```

This trains both models, prints performance metrics and cost-optimization
results, and opens 7 matplotlib/seaborn charts (ROC curve, delay
distribution, cost comparison, etc. — static copies of each are checked
into the repo root as `Screenshot 2025-10-23 *.png`; see Charts below).

## What it does

1. **Feature engineering** — lead time, planned transit days, weekday/month of departure, and a rolling per-carrier delay rate (see "Bug found and fixed" below).
2. **Two models**, both `RandomForest` (`sklearn.ensemble`), trained on an 80/20 chronological split:
   - `RandomForestClassifier` → probability a shipment is delayed (ROC-AUC).
   - `RandomForestRegressor` → expected delay in days (MAE).
3. **Cost optimization** on the most recent 300 shipments: compares the cost of expediting a shipment against its expected stockout cost (`delay probability × quantity × stockout cost per tonne`), and recommends expediting whichever is cheaper.

## Model performance (honestly reported)

| Metric | Before fix (leaky) | After fix |
|---|---|---|
| Delay Classifier ROC-AUC | 0.555 | **0.509** |
| Delay Regressor MAE | 2.25 days | **2.28 days** |

Measured directly from `python main.py yarn_supplychain_surat.csv`
(`RandomForest(random_state=42)`, so these are reproducible, not cherry-picked).

**This is a meaningfully worse headline number than earlier versions of
this README claimed (~0.82 AUC / ~1.4 days MAE), and that's the point —
see below.**

### Bug found and fixed: target leakage in `carrier_delay_30`

The rolling per-carrier delay-rate feature was computed as:

```python
df["carrier_delay_30"] = (
    df.groupby("carrier")["delay_flag"]
      .transform(lambda x: x.rolling(window=60, min_periods=1).mean())
)
```

`rolling()` includes the *current* row by default. That means each row's
feature partially encoded that same row's own label — the model was
reading a smeared copy of the answer. It's also named `_30` while using
`window=60`, a second, unrelated inconsistency.

Fixed to:

```python
df["carrier_delay_30"] = (
    df.groupby("carrier")["delay_flag"]
      .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
)
```

`shift(1)` excludes the current row, so the feature only reflects delay
history known *before* this shipment's outcome, and the window now
matches its name (30 of that carrier's most recent shipment records —
note this is a row count, not a calendar-day window; there's no date
resampling here).

**Once the leak was removed, ROC-AUC dropped from 0.555 to 0.509** — barely
above random (0.5). That's the honest result: on this synthetic dataset,
these features carry very little genuine predictive signal for
`delay_flag` once the model can no longer partially see its own label.
The pre-fix 0.555 itself already didn't match the ~0.82 previously claimed
here, which was likely never reproduced against this exact script/dataset
combination — I'm reporting what I actually measured, not preserving an
old headline number.

## Cost optimization (measured, both before and after the fix)

| | Before fix | After fix |
|---|---|---|
| Baseline expected cost | ₹3,756,144.17 | ₹3,867,961.82 |
| Optimized cost | ₹2,404,654.98 | ₹2,411,863.27 |
| Expected savings | ₹1,351,489.20 | ₹1,456,098.55 |
| Shipments expedited | 224 / 300 | 227 / 300 |

This part of the pipeline is far less sensitive to the leakage fix than
the classifier's ROC-AUC — the recommendation logic compares costs using
the regressor's continuous delay-day output and each shipment's actual
economics, not a hard classification threshold.

![Cost comparison chart](<Screenshot 2025-10-23 010428.png>)

## Charts

The script produces 7 charts total, checked into the repo root
(`Screenshot 2025-10-23 *.png`) since that's where they were originally
committed. Two worth calling out:

![Delay distribution](<Screenshot 2025-10-23 010354.png>)

*Most shipments arrive on time or early; the right tail (up to ~14 days)
is what the model is trying to predict.*

![Average delay by carrier](<Screenshot 2025-10-23 010406.png>)

The remaining three (`010415`: shipping cost vs. delay days, `010437`:
expedite decision counts, `010443`: predicted delay probability
distribution) are the same chart types the script generates each run.

> **About `Screenshot 2025-10-23 010303.png` (ROC curve):** this
> screenshot is from an earlier snapshot of the pipeline (before the
> leakage fix above, and before a later "refactor for clarity" commit)
> and shows AUC=0.635 — a third number, different from both figures in
> the table above. It's included to show what the script outputs, not as
> a current performance claim. Trust the **Model performance** table
> above over any number baked into an image.

## Tech stack

Python · pandas · NumPy · scikit-learn (`RandomForestClassifier`,
`RandomForestRegressor`) · matplotlib · seaborn

## Author

Raghav Malani
