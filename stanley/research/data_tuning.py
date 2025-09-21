import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import lightgbm as lgbm

train_data = pd.read_csv('stanley/data/train.csv')
test_data = pd.read_csv('stanley/data/test.csv')

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
feature_cols = [col for col in train_df if col not in ['time', 'Y1','Y2']]
input_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values
y1 = train_df['Y1'].values
y2 = train_df['Y2'].values

scale = StandardScaler()
train_scaled = scale.fit_transform(input_data)
test_scaled = scale.fit_transform(test_data)

pca = PCA(0.96, random_state=2025)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.fit_transform(test_scaled)

models = [
    ('rdg'      , Ridge(alpha=0.5)),
    ['ls'      : Lasso(alpha = 0.1)],
    ['en'     : ElasticNet(alpha = 0.1)],
    ['fr'     : RandomForestRegressor(n_estimators = 50, max_depth = None, random_state = 2025)],
    ['gb'  : GradientBoostingRegressor(n_estimators = 50, max_depth = None, random_state = 2025)],
    ['lgb'        : lgbm.LGBMRegressor(n_estimators = 300, learning_rate = 0.05, random_state = 2025)],
    ['nn'         : MLPRegressor(hidden_layer_sizes=(100,30))]
    ]

class Data_tuning:
    def stack_method(input_data, y1, y2, test_data):
        stack = StackingRegressor(
            estimators=models,
            final_estimator=Ridge(alpha = 0.5),
            passthrough=False,
            cv = 6
        )
        model_y1 = stack.fit(input_data, y1)
        model_y2 = stack.fit(input_data, y2)
        y1_pred = model_y1.predict(test_data)
        y2_pred = model_y2.predict(test_data)

        y1_r2 = r2_score(y1, model_y1)
        y2_r2 = r2_score(y2, model_y2)

        pd.DataFrame({'Y1' : y1_pred, 'Y2' : y2_pred}).to_csv('stanley/data/stack_pred.csv')
        print(f"Y1_R2 = {y1_r2}, Y2_R2 = {y2_r2}")
        return
    
Data_tuning.stack_method(input_data, y1, y2, test_data)


       

