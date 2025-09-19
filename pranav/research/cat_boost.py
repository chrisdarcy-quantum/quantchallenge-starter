import pandas as pd
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score


train_df = pd.read_csv('/workspaces/quantchallenge-starter/pranav/data/train.csv')
test_df = pd.read_csv('/workspaces/quantchallenge-starter/pranav/data/test.csv')

print(train_df.head())
print(test_df.head())



# train_df.to_pandas(use_pyarrow_extension_array=False)
print('STAGE 1\n\n\n\n')
def time_features(train_df, base_cols, target_cols=('Y1','Y2'), lags=(1,2,3,5,10,20), rolls=(93,5,10,20,50)):
    df_sorted = train_df.sort_values('time')
    
    for col in list(base_cols)+list(target_cols):
        for k in lags:
            df_sorted[f'{col}_lag{k}']=df_sorted[col].shift(k)
    
    def add_rolls(col):
        s = df_sorted[col].shift(1)
        for w in rolls:
            m = s.rolling(w, min_periods = max(2, int(0.6*w)))
            df_sorted[f'{col}_rmean{w}'] = m.mean()
            df_sorted[f'{col}_rstd{w}'] = m.std()
            df_sorted[f'{col}_rmin{w}'] = m.min()
            df_sorted[f'{col}_rmax{w}'] = m.max()
            
    for col in base_cols:
        add_rolls(col)
        
    
    for col in base_cols:
        df_sorted[f'{col}_diff1'] = df_sorted[col].diff(1)
        
    for P in (24, 48, 96, 256, 1024):
        df_sorted[f'sin_{P}'] = np.sin(2*np.pi*df_sorted['time']/P)
        df_sorted[f'cos_{P}'] = np.cos(2*np.pi*df_sorted['time']/P)
        
    df_sorted = df_sorted.dropna()
    
    return df_sorted


print('STAGE 2\n\n\n')


feature_cols_base = [c for c in train_df.columns if c not in ['time','Y1','Y2']]
df_fe = time_features(train_df, feature_cols_base, target_cols=('Y1','Y2'))

t_train_end, t_valid_end = 60000, 70000
train = df_fe[df_fe['time']<=t_train_end]
valid = df_fe[(df_fe['time']>t_train_end) & (df_fe['time'] <= t_valid_end)]
test = df_fe[df_fe['time']>t_valid_end]

x_cols = [c for c in df_fe.columns if c not in ['time', 'Y1', 'Y2']]
y_cols = ['Y1', 'Y2']

x_train, y1_train, y2_train = train[x_cols], train['Y1'], train['Y2']
x_valid, y1_valid, y2_valid = valid[x_cols], valid['Y1'], valid['Y2']
x_test, y1_test, y2_test = test[x_cols], test['Y1'], test['Y2']

pool_train_y1 = Pool(x_train, y1_train)
pool_valid_y1 = Pool(x_valid, y1_valid)
pool_test_y1 = Pool(x_test, y1_test)


pool_train_y2 = Pool(x_train, y2_train)
pool_valid_y2 = Pool(x_valid, y2_valid)
pool_test_y2 = Pool(x_test, y2_test)

cat_params = dict(loss_function = 'RMSE',
                  learning_rate=0.05,
                  depth=6,
                  l2_leaf_reg=5.0,
                  random_strength=1.0,
                  bootstrap_type='Bernoulli',
                  subsample=0.8,
                  rsm=0.8,
                  iterations=1000,
                  od_type='Iter',
                  od_wait=300,
                  random_seed=42,
                  verbose=False)


model_y1 = CatBoostRegressor(**cat_params)
model_y2 = CatBoostRegressor(**cat_params)


print('STAGE 3\n\n\n')

model_y1.fit(pool_train_y1, eval_set=pool_valid_y1, use_best_model=True, verbose=200)
model_y2.fit(pool_train_y2, eval_set=pool_valid_y2, use_best_model=True, verbose=200)

y1_pred_valid = model_y1.predict(pool_valid_y1)
y2_pred_valid = model_y2.predict(pool_valid_y2)
y1_pred_test = model_y1.predict(pool_test_y1)
y2_pred_test = model_y2.predict(pool_test_y2)

r2_valid_y1 = r2_score(y1_valid, y1_pred_valid)
r2_valid_y2 = r2_score(y2_valid, y2_pred_valid)
r2_test_y1 = r2_score(y1_test, y1_pred_test)
r2_test_y2 = r2_score(y2_test, y2_pred_test)

print(f'Valid R2 - Y1 {r2_valid_y1:.4f}, Y2:{r2_valid_y2:.4f}, Avg: {(r2_valid_y1+r2_valid_y2)/2:.4f}')
print(f'Test R2 - Y1 {r2_test_y1:.4f}, Y2:{r2_test_y2:.4f}, Avg: {(r2_test_y1+r2_test_y2)/2:.4f}')


def top_importances(model, cols, topn=20):
    imp = pd.Series(model.get_feature_importance(type='FeatureImportance'), index=cols)
    return imp.sort_values(ascending=False).head(topn)



print('Top Features for Y1:')
print(top_importances(model_y1, x_cols, 20))
print('Top Features for Y2:')
print(top_importances(model_y2, x_cols, 20))

preds = pd.DataFrame({
    'id':test_df['id'],
    'Y1':y1_pred_test,
    'Y2':y2_pred_test
})
preds['Y1'] = y1_pred_test.mean()
preds['Y2'] = y2_pred_test.mean()

preds.to_csv('preds.csv')

print(preds.head())
