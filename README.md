# QuantChallenge 2025 - Multi-Target Time Series Prediction

## Problem Overview

This is a **time-series forecasting challenge** focused on predicting two target variables (`Y1` and `Y2`) based on historical market-like features. The problem involves:

- **Input Features**: 14 continuous features (A through N) plus a time index
- **Target Variables**: 
  - `Y1`: Primary prediction target
  - `Y2`: Secondary prediction target (can be used as a feature for Y1)
- **Evaluation Metric**: R² (coefficient of determination) - average of R² scores for both targets

### Data Structure

**Training Data**: Time-series observations with features A-N and targets Y1, Y2
- Features are normalized continuous variables
- Time-ordered sequential data requiring temporal awareness
- Training set contains historical observations with known targets

**Test Data**: Future time periods with only features A-N (no Y1, Y2)
- Must predict both Y1 and Y2 for submission
- Same feature structure as training data

## Approach Implemented

### 1. Feature Engineering
The solution implements sophisticated feature engineering to capture temporal patterns:

- **Lagged Features**: Creates lag features for periods [1, 2, 3, 5, 10] for all input features
- **Rolling Statistics**: Computes rolling mean and standard deviation over windows [5, 20]
- **Cross-Feature Information**: Uses Y2 lags as predictive features

### 2. Multi-Model Strategy

#### Y1 Prediction (Ensemble Approach)
- **Ridge Regression**: Linear model with L2 regularization for stable predictions
- **XGBoost**: Gradient boosted trees to capture non-linear patterns
  - 200 estimators, max depth 4, learning rate 0.05
  - Subsample & column sampling at 0.8 for regularization
- **Ensemble**: 50/50 weighted average of Ridge and XGBoost predictions

#### Y2 Prediction (Time Series Approach)
- **SARIMAX (ARIMAX)**: AutoRegressive model with exogenous features
  - Order (1,0,0): AR(1) process
  - Uses Y2 lagged features as exogenous variables
- Chosen for its ability to model temporal dependencies directly

### 3. Validation Strategy
- **Time-based split**: 80/20 train/validation split
- Preserves temporal ordering (no random shuffling)
- Separate R² evaluation for both targets

### 4. Test Prediction Pipeline
1. Concatenate train + test data for consistent feature generation
2. Generate features across the full timeline
3. Forecast Y2 using ARIMAX model trained on historical data
4. Use forecasted Y2 to create lag features for test set
5. Predict Y1 using the trained ensemble model
6. Generate final submission file

## Project Structure

```
chris/
├── README.md                  # This file
├── data/
│   ├── train.csv             # Training data with Y1, Y2 targets
│   └── test.csv              # Test data (predict Y1, Y2)
└── research/
    ├── market_prediction.py   # Main prediction pipeline
    ├── submission.csv         # Generated predictions
    └── preds.csv             # Intermediate predictions
```

## Key Technical Challenges

1. **Temporal Leakage Prevention**: Must respect time ordering in splits and feature generation
2. **Cold Start Problem**: Test predictions require forecasting Y2 without ground truth
3. **Feature Propagation**: Ensuring lagged features align correctly across train/test boundary
4. **Model Selection**: Balancing between linear (Ridge) and non-linear (XGBoost) models
5. **Missing Data**: Handling NaN values from lag/rolling calculations

## Dependencies

```python
pandas, numpy          # Data manipulation
sklearn               # Ridge, preprocessing, metrics
xgboost              # Gradient boosting
statsmodels          # SARIMAX time series modeling
```

## Running the Code

```bash
cd chris/research
python market_prediction.py
```

This generates `submission.csv` with predictions for test set.

## Sylvian Extension Setup

Make sure you have installed the [Sylvian extension](https://marketplace.visualstudio.com/items?itemName=SylvianAI.sylvian) and initialized it. **This is required to be eligible for prizes!**

1. Go to the command palette (⇧⌘P on Mac, Ctrl + Shift + P otherwise)
2. Search for 'Sylvian: Initialize Sylvian'
3. Enter the email you used for the competition

If done correctly, your .competition file should include `email=your_email_here`. **DO NOT EDIT THIS .competition FILE**!

After having worked in your repository for a little, you should be able to go to quantchallenge.org > Dashboard > Settings and see that the extension is active. If it is not active, please contact support in the Discord!

## Questions & Support
If you have any lingering questions, reach out for support on Discord or email info@quantchallenge.org
