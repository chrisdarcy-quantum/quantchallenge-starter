import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

train_df = pd.DataFrame(pd.read_csv("stanley/data/train.csv"))
test_df = pd.DataFrame(pd.read_csv("stanley/data/test.csv"))
feature_cols = [col for col in train_df if col not in ['time', 'Y1','Y2']]
input_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values
y1 = train_df['Y1'].values
y2 = train_df['Y2'].values

class Models:

    def linear_model(input_data, y1, y2, test_data):

        linear_results = {}
        model = {
            'type' : LinearRegression(),
            'Ridge': Ridge(alpha=1.2),
            'Lasso': Lasso(alpha=0.10),
            'Net'  : ElasticNet(alpha=0.1, l1_ratio=0.4)
        }
        for name, item in model.items():

            model_y1 = type(item)(**item.get_params())
            model_y1.fit(input_data, y1)
            y1_model = model_y1.predict(input_data)
            r2_y1 = r2_score(y1, y1_model)
            y1_pred = model_y1.predict(test_data)

            model_y2 = type(item)(**item.get_params())
            model_y2.fit(input_data, y2)
            y2_model = model_y2.predict(input_data)
            r2_y2 = r2_score(y2, y2_model)
            y2_pred = model_y2.predict(test_data)

            pd.DataFrame({  'Y1' : y1_pred, 
                            'Y2' : y2_pred}).to_csv(f'stanley/data/likelihood_{name}_preds.csv')
            
            print(f'Y1 R2={r2_y1}, Y2 R2={r2_y2}')
            
            linear_results[name] = {
                'y1_model' : y1_model,
                'y2_model' : y2_model,
                'y1_r2'    : r2_y1,
                'y2_r2'    : r2_y2
            }
        return linear_results
    
    def polynomial_model(input_data, y1, y2, test_data):

        degrees = [2,3,4]
        polynomial_results = {}
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            input_poly = poly.fit_transform(input_data)
            test_poly = poly.fit_transform(test_data)

            model_y1 = Ridge(alpha=1)
            model_y2 = Ridge(alpha=1)

            model_y1.fit(input_poly,y1)
            model_y2.fit(input_poly,y2)

            y1_model = model_y1.predict(input_poly)
            y2_model = model_y2.predict(input_poly)

            r2_y1 = r2_score(y1, y1_model)
            r2_y2 = r2_score(y2, y2_model)

            y1_pred = model_y1.predict(test_poly)
            y2_pred = model_y2.predict(test_poly)

            pd.DataFrame({  'Y1' : y1_pred, 
                            'Y2' : y2_pred}).to_csv(f'stanley/data/polynomial_{degree}_preds.csv')
            
            polynomial_results[f'Polynomial_Degree_{degree}'] = {
                'y1_model' : y1_model,
                'y2_model' : y2_model,
                'y1_r2'    : r2_y1,
                'y2_r2'    : r2_y2
            }
            print(f'Y1 R2={r2_y1}, Y2 R2={r2_y2}')
        return polynomial_results
    
    def lost_in_the_forest(n, input_data, y1, y2, test_data):

        trees = {
            'Forest'    : RandomForestRegressor(n_estimators = n, random_state = 2025),
            'GradBoost' : GradientBoostingRegressor(n_estimators = n, random_state= 2025)
        }
        tree_results = {}
        for name, model in trees.items():
            model_y1 = type(model)(**model.get_params())
            model_y2 = type(model)(**model.get_params())

            model_y1.fit(input_data, y1)
            model_y2.fit(input_data, y2)

            y1_model = model_y1.predict(input_data)
            y2_model = model_y2.predict(input_data)

            r2_y1 = r2_score(y1, y1_model)
            r2_y2 = r2_score(y2, y2_model)

            y1_pred = model_y1.predict(test_data)
            y2_pred = model_y2.predict(test_data)

            pd.DataFrame({  'Y1' : y1_pred, 
                            'Y2' : y2_pred}).to_csv(f'stanley/data/{name}_preds.csv')            
            
            tree_results[name] = {
                'y1_model' : y1_model,
                'y2_model' : y2_model,
                'y1_r2'    : r2_y1,
                'y2_r2'    : r2_y2
            }

            print(f'Y1 R2={r2_y1}, Y2 R2={r2_y2}')
        return tree_results
    
    def caught_in_a_neural_net(input_data, y1, y2, test_data):

        scalar_value = StandardScaler()
        input_nn = scalar_value.fit_transform(input_data)
        test_nn = scalar_value.fit_transform(test_data)
        nn_results = {}

        nn_configs = {
            "Small" : (50, ),
            "Medium": (100,30),
            "Large" : (200, 60, 10)
        }

        for conf,layer in nn_configs.items():
            model_y1 = MLPRegressor(hidden_layer_sizes = layer, max_iter = 350, random_state = 2025)
            model_y2 = MLPRegressor(hidden_layer_sizes = layer, max_iter = 350, random_state = 2025)

            model_y1.fit(input_nn, y1)
            model_y2.fit(input_nn, y2)

            y1_model = model_y1.predict(input_nn)
            y2_model = model_y2.predict(input_nn)

            r2_y1 = r2_score(y1, y1_model)
            r2_y2 = r2_score(y2, y2_model)

            y1_pred = model_y1.predict(test_nn)
            y2_pred = model_y2.predict(test_nn)
            pd.DataFrame({  'Y1' : y1_pred, 
                            'Y2' : y2_pred}).to_csv(f'stanley/data/nn_{conf}_preds.csv')

            nn_results[conf] = {
                'y1_model' : y1_model,
                'y2_model' : y2_model,
                'y1_r2'    : r2_y1,
                'y2_r2'    : r2_y2
            }
            print(f'Y1 R2={r2_y1}, Y2 R2={r2_y2}')
        return nn_results
    
    def some_would_say_heavy(feature_columns, input_data, y1, y2, test_data, epsilon, n):
        heavy_input_features = []
        heavy_test_features = []
        kde_models = {}
        heavy_results = {}

        for i, col in enumerate(feature_cols):
            gauss = gaussian_kde(input_data[:,i])
            kde_models[col] = gauss
            heavy = gauss.evaluate(input_data[:,i])
            log = np.log(heavy + epsilon)
            heavy_input_features.append(log)
        input_heavy = heavy_input_features.T

        for i, col in enumerate(feature_cols):
            gauss = gaussian_kde(test_data[:,i])
            kde_models[col] = gauss
            heavy = gauss.evaluate(test_data[:,i])
            log = np.log(heavy + epsilon)
            heavy_test_features.append(log)
        test_heavy = heavy_test_features.T
        
        models = {
            'H_Linear' : LinearRegression(),
            'H_Ridge'  : Ridge(alpha = 1),
            'H_Forest' : RandomForestRegressor(n_estimators=n, random_state=2025)
        }

        for name, model in models.item():
            model_y1 = type(model)(**model.get_params())
            model_y2 = type(model)(**model.get_params())

            model_y1.fit(input_heavy, y1)
            model_y2.fit(input_heavy, y2)

            y1_model = model_y1.predict(input_heavy)
            y2_model = model_y2.predict(input_heavy)

            r2_y1 = r2_score(y1, y1_model)
            r2_y2 = r2_score(y2, y2_model)

            y1_pred = model_y1.predict(test_heavy)
            y2_pred = model_y2.predict(test_heavy)
            pd.DataFrame({  'Y1' : y1_pred, 
                            'Y2' : y2_pred}).to_csv(f'stanley/data/heavy_{name}_preds.csv')

            heavy_results[name] = {
                'y1_model' : y1_model,
                'y2_model' : y2_model,
                'y1_r2'    : r2_y1,
                'y2_r2'    : r2_y2
            }
            print(f'Y1 R2={r2_y1}, Y2 R2={r2_y2}')


Models.lost_in_the_forest(40, input_data, y1, y2, test_data)