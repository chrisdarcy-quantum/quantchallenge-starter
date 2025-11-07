import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Constants for consistent feature engineering
LAGS = [1, 2, 3, 5, 10]
ROLLINGS = [5, 20, 60]
BASE_FEATURES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']


def make_features(df, include_y2=True):
    """
    Create lagged features, rolling statistics, difference features, and z-score features.
    Uses efficient pd.concat to avoid DataFrame fragmentation.
    """
    out = df.copy()
    
    cols_to_process = BASE_FEATURES.copy()
    if include_y2 and 'Y2' in df.columns:
        cols_to_process.append('Y2')
    
    new_features = {}
    
    for col in cols_to_process:
        for lag in LAGS:
            new_features[f"{col}_lag{lag}"] = out[col].shift(lag)
    
    for col in BASE_FEATURES:
        for w in ROLLINGS:
            roll_mean = out[col].rolling(w).mean()
            roll_std = out[col].rolling(w).std()
            new_features[f"{col}_roll{w}_mean"] = roll_mean
            new_features[f"{col}_roll{w}_std"] = roll_std
            new_features[f"{col}_roll{w}_zscore"] = (out[col] - roll_mean) / (roll_std + 1e-8)
    
    top_y2_features = ['A', 'K', 'D', 'I', 'B']
    for col in top_y2_features:
        for span in [5, 20, 60]:
            new_features[f"{col}_ewm{span}_mean"] = out[col].ewm(span=span, adjust=False).mean()
            new_features[f"{col}_ewm{span}_std"] = out[col].ewm(span=span, adjust=False).std()
    
    for col in BASE_FEATURES:
        new_features[f"{col}_diff1"] = out[col].diff(1)
    
    if include_y2 and 'Y2' in df.columns:
        new_features[f"Y2_diff1"] = out['Y2'].diff(1)
    
    new_features_df = pd.DataFrame(new_features, index=out.index)
    out = pd.concat([out, new_features_df], axis=1)
    
    return out


print("Loading training data...")
train = pd.read_csv("data/train.csv").sort_values("time").reset_index(drop=True)
    

print("Creating training features...")
train_feat = make_features(train, include_y2=True)
train_feat = train_feat.replace([np.inf, -np.inf], np.nan)

print("Imputing missing values...")
train_feat_clean = train_feat.dropna().reset_index(drop=True)

X = train_feat_clean.drop(columns=['Y1', 'Y2'])
y1 = train_feat_clean['Y1']
y2 = train_feat_clean['Y2']

# Time split 80/20
split = int(0.8 * len(train_feat_clean))
X_train, X_val = X.iloc[:split], X.iloc[split:]
y1_train, y1_val = y1.iloc[:split], y1.iloc[split:]
y2_train, y2_val = y2.iloc[:split], y2.iloc[split:]

print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")


print("\nTraining Y2 model (XGBoost with A-N features only)...")

y2_feature_cols = [col for col in X_train.columns if not col.startswith('Y2_')]
X_train_y2 = X_train[y2_feature_cols]
X_val_y2 = X_val[y2_feature_cols]

imputer_y2 = SimpleImputer(strategy="median")
X_train_y2_imputed = imputer_y2.fit_transform(X_train_y2)
X_val_y2_imputed = imputer_y2.transform(X_val_y2)

xgb_y2 = XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=3,
    random_state=42,
    early_stopping_rounds=50
)

xgb_y2.fit(
    X_train_y2_imputed,
    y2_train,
    eval_set=[(X_val_y2_imputed, y2_val)],
    verbose=False
)

pred_y2 = xgb_y2.predict(X_val_y2_imputed)

r2_y2_val = r2_score(y2_val, pred_y2)
print(f"Y2 Validation R²: {r2_y2_val:.4f}")


print("\nTraining Y1 models...")

ridge_y1 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

ridge_y1.fit(X_train, y1_train)
pred_ridge_y1_val = ridge_y1.predict(X_val)

huber_y1 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=200))
])

huber_y1.fit(X_train, y1_train)
pred_huber_y1_val = huber_y1.predict(X_val)

xgb_y1 = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)

imputer_y1 = SimpleImputer(strategy="median")
X_train_imputed = imputer_y1.fit_transform(X_train)
X_val_imputed = imputer_y1.transform(X_val)

xgb_y1.fit(
    X_train_imputed, 
    y1_train,
    eval_set=[(X_val_imputed, y1_val)],
    verbose=False
)

pred_xgb_y1_val = xgb_y1.predict(X_val_imputed)

print("\nLearning optimal ensemble weights...")
blend_features = np.column_stack([pred_ridge_y1_val, pred_huber_y1_val, pred_xgb_y1_val])
blend_model = LinearRegression(fit_intercept=False, positive=True)
blend_model.fit(blend_features, y1_val)
blend_weights = blend_model.coef_
blend_weights = blend_weights / blend_weights.sum()

print(f"Learned blend weights - Ridge: {blend_weights[0]:.3f}, Huber: {blend_weights[1]:.3f}, XGBoost: {blend_weights[2]:.3f}")

pred_y1_val = (blend_weights[0] * pred_ridge_y1_val + 
               blend_weights[1] * pred_huber_y1_val + 
               blend_weights[2] * pred_xgb_y1_val)

r2_y1_val = r2_score(y1_val, pred_y1_val)
print(f"Y1 Validation R²: {r2_y1_val:.4f}")

r2_final = 0.5 * (r2_y1_val + r2_y2_val)
print(f"\nFinal average R²: {r2_final:.4f}")



print("\n" + "="*50)
print("Generating test predictions...")
print("="*50)

test = pd.read_csv("data/test.csv").sort_values("time").reset_index(drop=True)

train_cols = ['time'] + BASE_FEATURES
if 'Y1' in train.columns:
    train_for_concat = train[train_cols + ['Y1', 'Y2']]
else:
    train_for_concat = train[train_cols]

test_for_concat = test[['time'] + BASE_FEATURES]

both = pd.concat([train_for_concat, test_for_concat], ignore_index=True)

print("Creating base features for combined data...")
both_feat = both.copy()

new_features = {}

for col in BASE_FEATURES:
    for lag in LAGS:
        new_features[f"{col}_lag{lag}"] = both_feat[col].shift(lag)

for col in BASE_FEATURES:
    for w in ROLLINGS:
        roll_mean = both_feat[col].rolling(w).mean()
        roll_std = both_feat[col].rolling(w).std()
        new_features[f"{col}_roll{w}_mean"] = roll_mean
        new_features[f"{col}_roll{w}_std"] = roll_std
        new_features[f"{col}_roll{w}_zscore"] = (both_feat[col] - roll_mean) / (roll_std + 1e-8)

top_y2_features = ['A', 'K', 'D', 'I', 'B']
for col in top_y2_features:
    for span in [5, 20, 60]:
        new_features[f"{col}_ewm{span}_mean"] = both_feat[col].ewm(span=span, adjust=False).mean()
        new_features[f"{col}_ewm{span}_std"] = both_feat[col].ewm(span=span, adjust=False).std()

for col in BASE_FEATURES:
    new_features[f"{col}_diff1"] = both_feat[col].diff(1)

new_features_df = pd.DataFrame(new_features, index=both_feat.index)
both_feat = pd.concat([both_feat, new_features_df], axis=1)

print("\nPredicting Y2 for test set...")
test_feat_y2 = both_feat.iloc[len(train):].copy()
test_feat_y2 = test_feat_y2[y2_feature_cols]
test_feat_y2 = test_feat_y2.replace([np.inf, -np.inf], np.nan)

test_feat_y2_imputed = imputer_y2.transform(test_feat_y2)

y2_test_forecast = xgb_y2.predict(test_feat_y2_imputed)
y2_test_forecast = pd.Series(y2_test_forecast)

y2_train_series = train['Y2'].reset_index(drop=True)
y2_full = pd.concat([y2_train_series, y2_test_forecast.reset_index(drop=True)], ignore_index=True)

y2_features = {}
for lag in LAGS:
    y2_features[f"Y2_lag{lag}"] = y2_full.shift(lag)
y2_features["Y2_diff1"] = y2_full.diff(1)

y2_features_df = pd.DataFrame(y2_features, index=both_feat.index)
both_feat = pd.concat([both_feat, y2_features_df], axis=1)

test_feat = both_feat.iloc[len(train):].copy()
test_feat = test_feat.replace([np.inf, -np.inf], np.nan)

train_features_cols = X_train.columns.tolist()

missing = [c for c in train_features_cols if c not in test_feat.columns]
if missing:
    print(f"Warning: {len(missing)} features missing in test set, will be imputed")
    for c in missing:
        test_feat[c] = np.nan

test_feat = test_feat[train_features_cols]

print("\nPredicting Y1 for test set...")
pred_test_y1_ridge = ridge_y1.predict(test_feat)
pred_test_y1_huber = huber_y1.predict(test_feat)

test_feat_imputed = imputer_y1.transform(test_feat)
pred_test_y1_xgb = xgb_y1.predict(test_feat_imputed)

pred_test_y1 = (blend_weights[0] * pred_test_y1_ridge + 
                blend_weights[1] * pred_test_y1_huber + 
                blend_weights[2] * pred_test_y1_xgb)

pred_test_y2 = y2_test_forecast.values

submission = pd.DataFrame({
    "id": test["id"],
    "Y1": pred_test_y1,
    "Y2": pred_test_y2
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file created: submission.csv")
print("Done!")
