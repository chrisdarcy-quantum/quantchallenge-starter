import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Load + Feature Engineering

train = pd.read_csv("data/train.csv").sort_values("time").reset_index(drop=True)

def make_features(df, lags=[1,2,3,5,10], rollings=[5,20]):
    out = df.copy()

    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Y2']:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']:
        for w in rollings:
            out[f"{col}_roll{w}_mean"] = out[col].rolling(w).mean()
            out[f"{col}_roll{w}_std"] = out[col].rolling(w).std()

    return out
    

train_feat = make_features(train)
train_feat = train_feat.replace([np.inf, -np.inf], np.nan)
train_feat = train_feat.dropna().reset_index(drop=True)

X = train_feat.drop(columns=['Y1', 'Y2'])
y1 = train_feat['Y1']
y2 = train_feat['Y2']


# Time split 80/20
split = int(0.8 * len(train_feat) )
X_train, X_val = X.iloc[:split], X.iloc[split:]
y1_train, y1_val = y1.iloc[:split], y1.iloc[split:]
y2_train, y2_val = y2.iloc[:split], y2.iloc[split:]



# Y1 Models: Ridge + XGB


ridge_y1 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
    ])

ridge_y1.fit(X_train, y1_train)
pred_ridge_y1 = ridge_y1.predict(X_val)

xgb_y1 = XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

xgb_y1.fit(X_train, y1_train)
pred_xgb_y1 = xgb_y1.predict(X_val)


# Blend
pred_y1 = 0.5*pred_ridge_y1 + 0.5*pred_xgb_y1
print("R^2 Y1:", r2_score(y1_val, pred_y1))
      

# Y2 model: ARIMAX (with exogenous features)

exog_train = X_train[['Y2_lag1', 'Y2_lag2', 'Y2_lag5', 'Y2_lag10']].fillna(0)
exog_val = X_val[['Y2_lag1', 'Y2_lag2', 'Y2_lag5', 'Y2_lag10']].fillna(0)

arimax = SARIMAX(y2_train, order=(1,0,0), exog=exog_train)
res = arimax.fit(disp=False)
pred_y2 = res.predict(start=len(y2_train), end=len(y2_train)+len(y2_val)-1, exog=exog_val)

print("R^2 Y2:", r2_score(y2_val, pred_y2))


# Stacked outputs

r2_final = 0.5*(r2_score(y1_val, pred_y1) + r2_score(y2_val, pred_y2))
print("Final average R^2:", r2_final)



# Now we go ahead and use test data
test = pd.read_csv("data/test.csv").sort_values("time").sort_values("time").reset_index(drop=True)

train_cols = train.columns

both = pd.concat( [train[train_cols], test[['time', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']]],
                    ignore_index=True)


def make_features_AN_only(df, lags=(1,2,3,5,10), rollings=(5,20)):
    out = df.copy()
    base = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

    for c in base:
        for lag in lags:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)

    for c in base:
        for w in rollings:
            r = out[c].rolling(w)
            out[f"{c}_roll{w}_mean"] = r.mean()
            out[f"{c}_roll{w}_std"] = r.std()

    return out


both_feat = make_features_AN_only(both)

h = len(test)
y2_train_series = train['Y2'].reset_index(drop=True)

ar1 = SARIMAX(y2_train_series, order=(1,0,0))
ar1_res = ar1.fit(disp=False)
y2_test_forecast = ar1_res.get_forecast(steps=h).predicted_mean


y2_full = pd.concat([y2_train_series, y2_test_forecast.reset_index(drop=True)], ignore_index=True)

for lag in [1,2,5,10]:
    both_feat[f"Y2_lag{lag}"] = y2_full.shift(lag)


test_feat = both_feat.iloc[len(train):].copy()
test_feat = test_feat.replace([np.inf, -np.inf], np.nan)

train_features_cols = X_train.columns.tolist()

missing = [ c for c in train_features_cols if c not in test_feat.columns]
for c in missing:
    test_feat[c] = 0.0


test_feat = test_feat[train_features_cols]


# now we do predicted

pred_test_y1_ridge = ridge_y1.predict(test_feat)
pred_test_y1_xgb = xgb_y1.predict(test_feat)
pred_test_y1 = 0.5*pred_test_y1_ridge + 0.5*pred_test_y1_xgb

pred_test_y2 = y2_test_forecast.values

submission = pd.DataFrame({
    "id": test["id"],
    "Y1": pred_test_y1,
    "Y2": pred_test_y2
})

submission.to_csv("submission.csv", index=False)
print("done")