# Predict Future Sales - Kaggle Competition

> [!NOTE]
> **Project Context**: This project was originally completed several years ago as part of a learning phase during a forecasting project at a former employer. The repository has been recently updated, restructured, and documented for review and showcase purposes.

## 🎯 Competition Overview

**Competition Link:** [Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales)

This competition is the final project for the "How to win a data science competition" Coursera course. The challenge is to predict total sales for every product and store in the next month based on historical sales data. 

### Dataset
- **Source:** Daily sales data from a Russian software company
- **Time Period:** January 2013 to October 2015
- **Task:** Predict sales for November 2015
- **Features:** 
  - `shop_id` - Unique identifier for shops
  - `item_id` - Unique identifier for products
  - `item_category_id` - Product category
  - `item_price` - Current price of item
  - `item_cnt_day` - Number of products sold per day

## 🔬 Approaches Implemented

### 1. XGBoost with Extensive Feature Engineering
**File:** `Predict-Sales.py`

This approach focuses on comprehensive feature engineering and gradient boosting.

**Key Features:**
- **Lag Features:** Created lag features for 1, 2, 3, 6, and 12 months
- **Mean Encoding:** 
  - Date-level averages
  - Shop-level averages
  - Item-level averages
  - Category-level averages
  - City-level averages
  - Combined features (shop-category, shop-type, item-city, etc.)
- **Trend Features:**
  - Price trends over 6 months
  - Revenue trends
  - Delta price calculations
- **Temporal Features:**
  - Month of year
  - Days in month
  - Time since last sale
  - Time since first sale
- **Data Preprocessing:**
  - Outlier removal (price > 100,000, sales > 1,000)
  - Shop consolidation (merging duplicate shops)
  - Category encoding (type and subtype extraction)

**Model Configuration:**
```python
XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3
)
```

**Validation Strategy:** 
- Train: Months 12-32
- Validation: Month 33
- Test: Month 34

### 2. LSTM Neural Network - Daily Aggregation
**Files:** 
- `Predict-Sales_RNN.py`
- `Predict-Sales_RNN_daily_train.py`
- `Predict-Sales_RNN_daily_train_v2.py`

Deep learning approach using LSTM for time series forecasting with daily data.

**Architecture:**
- Input: 29-day sliding window
- LSTM layers with dropout (0.2)
- MinMax scaling for normalization
- Time series to supervised learning transformation

**Key Techniques:**
- Series to supervised conversion
- 40 epochs training
- Batch size: 256
- Train/validation split: 60/40

### 3. LSTM Neural Network - Monthly Aggregation
**Files:**
- `Predict-Sales_RNN_monthly_train.py`
- `Predict-Sales_RNN_monthly_train_v2.py`
- `Predict-Sales_RNN_monthly_train_v3-Adding_ShopFeat.py`
- `Predict-Sales_RNN_monthly_train_v3-Adding_ShopFeat_v1.py` (Most advanced)

Enhanced LSTM approach with monthly aggregation and additional features.

**Evolution:**
- **v1:** Basic monthly aggregation
- **v2:** Improved data preprocessing
- **v3:** Added shop_id as feature
- **v3-v1:** Complete feature set with shop and block features

**Architecture (v3-v1):**
```python
Sequential([
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(2)  # Predicts both price and count
])
```

**Features:**
- Window size: 10 months
- Lag: 1 month ahead
- StandardScaler normalization
- Predicts both `item_price` and `item_cnt_day`
- Custom NaN filling strategy for missing prices
- Shop and item pair tracking

**Training Configuration:**
- Epochs: 10
- Batch size: 64
- Optimizer: Adam
- Loss: MSE

## 📊 Submissions & Results

Multiple submission files were generated through iterative improvements:

| Submission File | Approach | Key Features |
|----------------|----------|--------------|
| `xgb_submission.csv` | XGBoost | Full feature engineering |
| `Ali_submission.csv` | LSTM Daily | Basic daily model |
| `Ali_submission_daily_1.csv` | LSTM Daily | Improved daily model |
| `Ali_submission_daily_2_window1.csv` | LSTM Daily | Window size 1 |
| `Ali_submission3_monthly_window1.csv` | LSTM Monthly | Window size 1 |
| `Ali_submission3_monthly_window1_1Epoch.csv` | LSTM Monthly | Single epoch |
| `Ali_submission4_monthly_window3_addedFeatureShopId.csv` | LSTM Monthly | Window 3 + shop features |
| `Ali_submission_monthly_with_shopID_feat_1.csv` | LSTM Monthly | Shop ID features |

## 🔑 Key Learnings

### Feature Engineering
1. **Lag features are crucial** for time series - using multiple lag periods (1, 2, 3, 6, 12 months) captures different temporal patterns
2. **Mean encoding** at various aggregation levels (shop, item, category, city) provides powerful features
3. **Trend features** (price changes, revenue deltas) help capture market dynamics
4. **Temporal features** (months since last/first sale) encode product lifecycle information

### Data Preprocessing
1. **Outlier handling** is essential - removed extreme prices and sales counts
2. **Shop consolidation** - identified and merged duplicate shops
3. **Missing value strategy** - forward-filling prices by item, then by shop-item pairs
4. **Creating complete dataset** - generating all shop-item combinations for each month to match test set structure

### Model Selection
1. **XGBoost** excels with engineered features and handles missing data well
2. **LSTM** can capture temporal dependencies but requires careful preprocessing
3. **Monthly aggregation** performed better than daily for this problem
4. **Window size matters** - experimented with windows of 1, 3, and 10 months

### Neural Network Insights
1. **Dropout regularization** (0.2) helps prevent overfitting
2. **StandardScaler** worked better than MinMaxScaler for this dataset
3. **Predicting multiple outputs** (price and count) in one model is feasible
4. **Batch size and epochs** require tuning - larger batches (256) for daily, smaller (64) for monthly

## 🗂️ File Structure

```
Predict Sale Comp/
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
├── data/
│   ├── raw/                           # Original competition data
│   │   ├── sales_train.csv           # Training data (Jan 2013 - Oct 2015)
│   │   ├── test.csv                  # Test data (Nov 2015)
│   │   ├── items.csv                 # Item metadata
│   │   ├── item_categories.csv       # Category metadata
│   │   ├── shops.csv                 # Shop metadata
│   │   └── sample_submission.csv     # Submission template
│   └── processed/                     # Generated/processed datasets
│       ├── complete_train_set.csv    # Complete training set
│       ├── final_predicted_test_set.csv
│       └── data.pkl                  # Processed data for XGBoost
├── scripts/
│   ├── xgboost/
│   │   └── predict_sales_xgboost.py  # XGBoost with feature engineering
│   ├── lstm_daily/
│   │   ├── predict_sales_rnn.py      # Basic LSTM
│   │   ├── predict_sales_rnn_daily_v1.py
│   │   └── predict_sales_rnn_daily_v2.py
│   └── lstm_monthly/
│       ├── predict_sales_rnn_monthly_v1.py
│       ├── predict_sales_rnn_monthly_v2.py
│       ├── predict_sales_rnn_monthly_v3.py
│       └── predict_sales_rnn_monthly_v3_1.py  # Most advanced
├── models/
│   ├── lstm_daily/
│   │   ├── model_lstm_trained_daily
│   │   ├── model_lstm_trained_daily_window1
│   │   └── model_lstm_trained_daily_window3
│   └── lstm_monthly/
│       └── model_lstm_trained_monthly_window2_featuresShpIdBlkId_1Epoch
├── submissions/
│   ├── xgb_submission.csv            # XGBoost submission
│   ├── lstm_daily/
│   │   ├── Ali_submission_daily_1.csv
│   │   └── Ali_submission_daily_2_window1.csv
│   ├── lstm_monthly/
│   │   ├── Ali_submission.csv
│   │   ├── Ali_submission3_monthly_window1.csv
│   │   ├── Ali_submission3_monthly_window1_1Epoch.csv
│   │   ├── Ali_submission4_monthly_window3_addedFeatureShopId.csv
│   │   └── Ali_submission_monthly_with_shopID_feat_1.csv
│   └── training_results/
│       ├── result_trained_monthly_window10_full_trainSet_featuresShpIdBlkId_2Epoch64batch.csv
│       ├── trained_monthly_win10_full_10Ep256ba.csv
│       └── trained_monthly_win5_full_1Ep64ba.csv
└── outputs/
    └── temp-plot.html                # Visualization outputs
```

## 🚀 How to Run

### XGBoost Approach
```bash
cd scripts/xgboost
python predict_sales_xgboost.py
```
Requires: Data files in `data/raw/`

### LSTM Monthly (Latest Version)
```bash
cd scripts/lstm_monthly
python predict_sales_rnn_monthly_v3_1.py
```
Requires: `sales_train.csv`, `test.csv`

## 📦 Dependencies

```python
numpy
pandas
scikit-learn
xgboost
keras / tensorflow
matplotlib
seaborn
plotly
```

## 🎓 Competition Context

This competition is part of the Coursera course "How to win a data science competition: Learn from top Kagglers" and focuses on:
- Practical feature engineering techniques
- Handling time series data
- Model validation strategies
- Ensemble methods
- Dealing with real-world messy data

## 📝 Notes

- Target values are clipped to [0, 20] range as per competition requirements
- The dataset contains Russian shop names and categories
- Some items in test set are new (not in training data)
- Shop consolidation was necessary due to duplicate/renamed shops
- Monthly aggregation proved more stable than daily predictions

---

**Competition Status:** Completed  
**Best Approach:** XGBoost with extensive feature engineering  
**Key Takeaway:** Feature engineering often outperforms complex models in tabular time series data
