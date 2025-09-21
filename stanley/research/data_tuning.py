import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
import lightgbm as lgbm
import xgboost as xgb

train_data = pd.read_csv('stanley/data/train.csv')
test_data = pd.read_csv('stanley/data/test.csv')
extra_train_df = pd.DataFrame(pd.read_csv('stanley/data/train_new.csv'))
extra_test_df = pd.DataFrame(pd.read_csv('stanley/data/test_new.csv'))

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
feature_cols = [col for col in train_df if col not in ['time', 'Y1','Y2']]

include_extra_train = train_df[feature_cols].copy()
include_extra_train['O'] = extra_train_df['O'].fillna(0)
include_extra_train['P'] = extra_train_df['P'].fillna(0)

include_extra_test = test_df[feature_cols].copy()
include_extra_test['O'] = extra_test_df['O'].fillna(0)
include_extra_test['P'] = extra_test_df['P'].fillna(0)

extra_train_data = include_extra_train.values
extra_test_data = include_extra_test.values

input_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values
y1 = train_df['Y1'].values
y2 = train_df['Y2'].values

scale = StandardScaler()
train_scaled = scale.fit_transform(input_data)
test_scaled = scale.transform(test_data)

pca = PCA(0.96, random_state=2025)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)


class Model_Refining:
    def tune_a_ridge(input_data, y1, y2):
        params = {
            'alpha'    : [0.001, 0.01, 0.1, 1, 5, 10],
        }
        elastic = Ridge(random_state = 2025)
        map_y1 = RandomizedSearchCV(elastic, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = 1)
        map_y2 = RandomizedSearchCV(elastic, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = 1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_lasso(input_data, y1, y2):
        params = {
            'alpha'    : [0.0001, 0.001, 0.01, 0.1, 1, 5],
        }
        elastic = Lasso(random_state = 2025)
        map_y1 = RandomizedSearchCV(elastic, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = 1)
        map_y2 = RandomizedSearchCV(elastic, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = 1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_elastic_net(input_data, y1, y2):
        params = {
            'alpha'    : [0.001, 0.01, 0.1, 1, 5],
            'l1_ratio' : [0.1, 0.5, 0.8]
        }
        elastic = ElasticNet(random_state = 2025)
        map_y1 = RandomizedSearchCV(elastic, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = 1)
        map_y2 = RandomizedSearchCV(elastic, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = 1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_forest(input_data, y1, y2):
        params = {
            'n_estimators'     : [25, 50, 75],
            'max_depth'        : [10, 15],
            'min_samples_split' : [0.5, 3],
            'max_features'     : [2, 0.5]
        }
        forest = RandomForestRegressor(random_state = 2025)
        map_y1 = RandomizedSearchCV(forest, params, n_iter = 8, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y2 = RandomizedSearchCV(forest, params, n_iter = 8, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_gradBoost(input_data, y1, y2):
        params = {
            'n_estimators'     : [25, 50, 75],
            'max_depth'        : [10, 15],
            'min_samples_split' : [0.5, 3],
            'max_features'     : [2, 0.5]
        }
        forest = GradientBoostingRegressor(random_state = 2025)
        map_y1 = RandomizedSearchCV(forest, params, n_iter = 8, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y2 = RandomizedSearchCV(forest, params, n_iter = 8, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_light(input_data, y1, y2):
        params = {
            'n_estimators'     : [100, 400],
            'learning_rate'    : [0.05],
            'num_leaves'       : [75],
            'max_depth'        : [10, 20],
            'subsample'       : [0.5, 1],
            'colsample_bytree' : [0.5, 1]
        }
        light = lgbm.LGBMRegressor(random_state = 2025)
        map_y1 = RandomizedSearchCV(light, params, n_iter = 8, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y2 = RandomizedSearchCV(light, params, n_iter = 8, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_neural_net(input_data, y1, y2):
        params = {
            'hidden_layer_sizes' : [(100,30)],
            'alpha'             : [0.0005, 0.001, 0.01],
            'learning_rate_init': [0.01, 0.1],
            'max_iter'          : [70, 140]

        }
        nn = MLPRegressor(random_state = 2025)
        map_y1 = RandomizedSearchCV(nn, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y2 = RandomizedSearchCV(nn, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_a_cat(input_data, y1, y2):
        params = {
            'iterations'          : [100,300,1000],
            'learning_rate'       : [0.05, 0.1],
            'depth'               : [8],
            'bagging_temperature' : [0.5, 1],
            'l2_leaf_reg'         : [1, 3, 5]
        }
        cb = CatBoostRegressor(random_state = 2025)
        map_y1 = RandomizedSearchCV(cb, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y2 = RandomizedSearchCV(cb, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
    def tune_svr(input_data, y1, y2):
        params = {
            'C'       : [0.1, 1, 6],
            'epsilon' : [0.05, 0.1],
            'kernel'  : ['rbf'],
            'gamma'   : ['scale', 0.01, 0.1],
        }
        sr = SVR()
        map_y1 = RandomizedSearchCV(sr, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y2 = RandomizedSearchCV(sr, params, n_iter = 15, cv = 6, scoring = 'r2', n_jobs = -1)
        map_y1.fit(input_data, y1)
        map_y2.fit(input_data, y2)
        best_y1 = map_y1.best_params_
        best_y2 = map_y2.best_params_
        best_r2_y1 = map_y1.best_score_
        best_r2_y2 = map_y2.best_score_

        print(f"Best params Y1 = {best_y1} with score {best_r2_y1}")
        print(f"Best params Y2 = {best_y2} with score {best_r2_y2}")
        return [map_y1.best_estimator_, map_y2.best_estimator_]
class Tuner(Model_Refining):
    def stack_method(input_data, y1, y2, test_data):
        r = Model_Refining.tune_a_ridge(input_data, y1, y2)
        f = Model_Refining.tune_a_forest(input_data, y1, y2)
        g = Model_Refining.tune_a_gradBoost(input_data, y1, y2)
        lg = Model_Refining.tune_a_light(input_data, y1, y2)
        n = Model_Refining.tune_a_neural_net(input_data, y1, y2)
        xg_b = [xgb.XGBRegressor(n_estimators=300, subsample=1, max_depth=6, learning_rate=0.01, colsample_bytree=1, random_state = 2025), xgb.XGBRegressor(n_estimators=500, subsample=0.8, max_depth=10, learning_rate=0.01, colsample_bytree=0.8, random_state = 2025)]
        c = Model_Refining.tune_a_cat(input_data, y1, y2)
        stack_y1 = [
        ('rdg', r[0]),
        ('fr', f[0]),
        ('gb', g[0]),
        ('xgb', xg_b[0]),
        ('lgb', lg[0]),
        ('nn', n[0]),
        ('cb', c[0]),
        ]
        stack_y2 = [
        ('rdg', r[1]),
        ('fr', f[1]),
        ('gb', g[1]),
        ('xgb', xg_b[1]),
        ('lgb', lg[1]),
        ('nn', n[1]),
        ('cb', c[1])
        ]
        final_esti = Pipeline([
            ('scaler' , StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.1,0.5,1,2,5,10]))
        ])
        c_v = KFold(n_splits=5, shuffle=True, random_state=2025)

        y1_stack = StackingRegressor(
            estimators=stack_y1,
            final_estimator=final_esti,
            passthrough=False,
            cv = c_v,
            n_jobs=-1
        )

        y2_stack = StackingRegressor(
            estimators=stack_y2,
            final_estimator=final_esti,
            passthrough=False,
            cv = c_v,
            n_jobs=-1
        )

        cv_score_y1 = cross_val_score(y1_stack, input_data, y1, cv = c_v)
        cv_score_y2 = cross_val_score(y2_stack, input_data, y2, cv = c_v)

        model_y1 = y1_stack.fit(input_data, y1)
        model_y2 = y2_stack.fit(input_data, y2)

        y1_pred = model_y1.predict(test_data)
        y2_pred = model_y2.predict(test_data)


        preds = pd.DataFrame({'Y1' : y1_pred, 'Y2' : y2_pred})
        preds.to_csv(f'stanley/data/stack_pred_{input_data}.csv')
        print(f"Y1_R2 = {cv_score_y1.mean()} +- {cv_score_y1.std()}, Y2_R2 = {cv_score_y2.mean()} +- {cv_score_y2.std()}")
    
        return preds

Tuner.stack_method(input_data, y1, y2, test_data)
print('Now for the extra stuff!')
Tuner.stack_method(extra_train_data, y1, y2, extra_test_data)

       

