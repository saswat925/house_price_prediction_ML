# 🏠 House Price Prediction: End-to-End ML Project

An end-to-end machine learning project on a **1 lakh (101,200 rows × 26 columns)** Indian housing dataset with deliberately messy data. It covers data cleaning, EDA, regression, classification, clustering, time-series forecasting and model deployment in a single notebook.

## 📌 What's Inside

| Part | What it does |
|---|---|
| **A. Data** | Load → audit → clean → EDA → feature engineering |
| **B. Regression** | Predict `price_lakh` using 10 models, ensemble blend, tuning, 5-fold CV, residual analysis, permutation importance |
| **C. Classification** | (1) Will the house sell within 30 days? (2) Is the price Low / Medium / High? |
| **D. Unsupervised** | K-Means market segments + PCA visualisation |
| **E. Time Series** | 12-month price index forecast: Naive vs Holt-Winters vs ARIMA |
| **F. Deployment** | Saved pipeline (`.pkl`) + a `predict_price()` function |

## 🧹 Data Cleaning Highlights

- Removed **1,200 duplicate rows**
- Standardised messy spellings (`Bangalore`/`Bengaluru`, `BBSR`/`Bhubaneswar`, `Bombay`/`Mumbai`, etc.)
- Fixed impossible values (10 sq.ft homes, negative ages, floor above top floor)
- Filled missing values with group medians instead of one global number
- Removed wrong prices using price per sq.ft vs the city median
- Kept **99,405 of 101,200 rows (98.2%)**

## 📊 Results

| Task | Best Model | Result |
|---|---|---|
| Price regression | CatBoost | MAPE **6.22%**, R² **0.984** |
| Final tuned model (LightGBM) | LightGBM + log-target | MAE **9.05 lakh**, R² **0.982**, MAPE **6.48%** |
| 5-fold cross-validation | LightGBM pipeline | R² **0.980 ± 0.0005** (very stable) |
| Sold within 30 days | Logistic Regression | ROC-AUC **0.771** |
| Price category (Low/Med/High) | Logistic Regression | Accuracy **93.9%**, no Low↔High mistakes |
| Market segments | K-Means | k = 2 (silhouette 0.19, weak natural grouping) |
| Price index forecast | ARIMA(1,1,1) + drift | MAPE **0.76%** vs 2.19% for naive |

## 💡 All Insights

### 1. Data quality
- **9 columns had missing values.** `builder_rating` was the worst (12.0%), followed by `facing` (8.1%), `dist_metro_km` (6.1%), `age_years` (5.1%), `furnishing` (4.1%), `balconies` (3.1%), `bathrooms` (2.0%), `size_sqft` (1.0%) and the target `price_lakh` (0.5%).
- **1,200 exact duplicates** were found (101,200 rows → 100,000 unique `property_id`s). Duplicates can leak across train and test and fake a high score.
- **The city column had 26 spellings for 10 cities** (for example `BANGALORE`, `Bombay`, `BBSR`, `Calcutta`, `New Delhi`).
- **Impossible values found:** 60 sizes, 40 bedroom counts, 50 ages and 30 floor numbers (floor above the top floor).
- **Plain IQR outlier detection flagged 7,186 prices (7.1%)** above 320.94 lakh, with the maximum at 8,888.75 lakh. Most of these are genuine expensive homes (for example Mumbai), so the project instead compared each price per sq.ft with the **city median** and removed only **82 clearly wrong rows (0.08%)**.

### 2. Exploratory analysis
- **Price is heavily right-skewed** (skewness 2.84). After a log transform it drops to 0.30, so the models predict `log(price)`.
- **Location is the biggest price driver.** Mumbai costs about **7x more per sq.ft** than Cuttack:

| City | Median price (lakh) | Median price per sq.ft (₹) |
|---|---|---|
| Mumbai | 325 | 26,513 |
| Delhi | 193 | 15,678 |
| Bengaluru | 124 | 10,006 |
| Pune | 112 | 9,095 |
| Chennai | 107 | 8,595 |
| Hyderabad | 102 | 8,298 |
| Kolkata | 83 | 6,726 |
| Ahmedabad | 69 | 5,535 |
| Bhubaneswar | 66 | 5,370 |
| Cuttack | 47 | 3,808 |

- **Size is the strongest numeric driver** (correlation with price 0.49), but the relationship fans out because every city has its own rate per sq.ft.
- **Bedrooms and size are highly correlated** (0.82), which is multicollinearity. Regularised and tree models handle it well.
- **Amenities matter:** a house with 4 amenities has a **31% higher** city-adjusted price level than one with none.
- **Furnishing matters moderately:** fully furnished homes carry a **10% premium** over unfurnished ones.
- **Facing direction has almost no effect** (0.9% spread between best and worst).
- **Prices rise about 4.1% per year** (2019 to 2024).

### 3. Regression (predicting price)

| Model | MAE (lakh) | R² | MAPE |
|---|---|---|---|
| CatBoost | 8.67 | 0.98 | **6.22%** |
| LightGBM | 9.21 | 0.98 | 6.61% |
| XGBoost | 9.42 | 0.98 | 6.73% |
| HistGradientBoosting | 9.44 | 0.98 | 6.73% |
| Linear Regression | 14.28 | 0.86 | 9.38% |
| Ridge | 14.28 | 0.86 | 9.38% |
| Elastic Net | 14.45 | 0.87 | 9.48% |
| Lasso | 14.72 | 0.86 | 9.68% |
| Random Forest | 18.32 | 0.94 | 13.63% |
| Decision Tree | 21.94 | 0.91 | 16.53% |

- **Boosting models win clearly** (6.2% to 6.7% error). They learn the interactions left over after the main effects.
- **Linear models are a strong baseline** (about 9.4% error) because log-price is almost additive.
- **No major overfitting:** train and test R² are close for every model.
- **Blending the top 3 models did not help** (6.31% vs 6.22% for CatBoost alone), because the three boosting models make very similar mistakes.
- **Tuning helped slightly:** LightGBM improved from 6.61% to 6.48% MAPE.
- **The final model is stable.** 5-fold CV gave R² 0.9801 ± 0.0005 and MAE 9.14 ± 0.14 lakh.
- **Final test result:** MAE 9.05 lakh, RMSE 15.79, R² 0.982, MAPE 6.48%. The median error is 5.33%, and **90% of predictions are within ±13.1%**.
- **No city is treated unfairly.** MAPE ranges only from 6.27% (Delhi, easiest) to 6.68% (Ahmedabad, hardest), with bias close to 0 everywhere.
- **Accuracy holds across price bands:** 6.79% for the cheapest 25%, 6.46%, 6.30%, and 6.37% for the costliest 25%.
- **Top 5 price drivers:** `city`, `size_sqft`, `locality`, `years_since_2019` and `age_years`.
- **Near-zero importance** (candidates to drop): `bath_per_bed`, `floor_ratio`, `total_floors`, `gym_flag`, `security_flag`, `swimming_pool_flag`, `month`, `owner_type`, `is_new`, `rating_missing`, `facing`, `lift_flag` and `balconies`. The individual amenity flags score low most likely because `amenity_score` already captures them.

### 4. Classification
**Will it sell within 30 days?**
- Classes are imbalanced: **64% not sold, 36% sold**.
- Best ROC-AUC is **0.771** (Logistic Regression, LightGBM and Random Forest are almost tied). Accuracy is 0.67.
- The model catches **80% of houses that do sell** (recall), but only 53% of its "will sell" predictions are correct (precision).
- **Asking price is the strongest driver of a quick sale.**
- Sell rate by price category: **Low 64.3%, Medium 47.7%, High 14.1%**. Cheaper homes sell much faster.
- Sell rate by owner type: Builder 38.2%, Owner 37.9%, **Agent 31.0%**.
- Selling speed has a large random component (buyer luck, negotiation), so a perfect score is not possible.

**Price category (Low / Medium / High)**
- Accuracy is **93.9%** (Logistic Regression) and 93.4% (LightGBM).
- There were **0 Low↔High mistakes**. All 1,308 errors were between neighbouring classes, right at the borders (for example a 49 vs 51 lakh house).

### 5. Clustering (K-Means + PCA)
- Silhouette scores were low for every k, and the best was **k = 2 (0.187)**. Prices change smoothly, so there are no sharp natural groups.
- The two segments split mostly by **size**: 40,200 larger homes (median 1,694 sq.ft, 143.8 lakh) and 59,205 smaller homes (median 1,030 sq.ft, 82.0 lakh). Price per sq.ft is almost the same in both (₹8,303 vs ₹8,095).
- Segments are a convenient grouping for strategy, not hard truth. Try k = 4 or 5 for finer segments.

### 6. Time-series forecasting
- The monthly price index covers **84 months** (Jan 2019 to Dec 2025), with the last 12 months held out as the test.
- **ARIMA(1,1,1) with drift and Holt exponential smoothing tied at 0.76% MAPE**, about **3x better than the naive guess** (2.19%). Models with a trend term win because prices rise steadily.
- The 12-month forecast moves the index from 1.146 to 1.178 (**+2.8%**), a little slower than the 4.1% yearly average seen in the EDA.

### 7. What-if predictions from the deployed model

| Scenario | Predicted price |
|---|---|
| Cuttack, CDA, 1,200 sq.ft 3BHK flat | ₹61.6 lakh |
| Mumbai, Powai, 1,200 sq.ft 3BHK flat | ₹361.3 lakh (about 5.9x Cuttack) |
| Bhubaneswar, Patia, 1,800 sq.ft 4BHK villa | ₹148.2 lakh |
| Same Cuttack flat, 20 years old | ₹51.6 lakh (about 16% lower) |
| Same Cuttack flat with gym and pool | ₹68.8 lakh (about 12% higher) |

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `statsmodels` · `joblib`

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost statsmodels joblib jupyter

# 3. Open the notebook
jupyter notebook house_price_1lakh_end_to_end.ipynb
```

Make sure `house_price_dataset_1lakh.csv` is in the same folder as the notebook. You can also run it directly in **Google Colab** by uploading the notebook and the CSV.

## 🔮 Example Prediction

```python
predict_price("Cuttack", "CDA", "Apartment", 1200, 3)
# → ₹61.6 lakh

predict_price("Mumbai", "Powai", "Apartment", 1200, 3)
# → ₹361.3 lakh
```

## 📁 Files

```
├── house_price_1lakh_end_to_end.ipynb   # Main notebook
├── house_price_dataset_1lakh.csv        # Dataset
├── house_price_model.pkl                # Saved model pipeline (created by the notebook)
└── README.md
```

## ⚠️ Limitations and Next Steps

- Validate with a **time-based split** (train on past years, test on the latest year) for real use.
- Selling speed is naturally noisy, so treat the probability as guidance only.
- Add real features such as schools, hospitals and GPS location.
- Try target encoding for `locality`, add SHAP explanations, and deploy with **FastAPI** or **Streamlit**.
- Retrain regularly, because price levels drift each year.

## 👤 Author

**Saswat**
