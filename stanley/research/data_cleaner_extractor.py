import pandas as pd
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import csv



train_data = pd.read_csv('stanley/data/train.csv')
train_df = pd.DataFrame(train_data)
test_data = pd.read_csv('stanley/data/test.csv')
test_df = pd.DataFrame(test_data)
variables = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
train_var_data = train_data[variables]
targets = ['Y1','Y2']
train_target_val = train_data[targets]

class Moving_average:
    ## construct a list of averages per time period
    def average_to_T(T, df):
        average = np.zeros(df.shape)
        id, cols = df.shape
        if T > id:
            print("Not Enough Rows, returning max size")
            T = id
        for x in range(T):
            for col in range(cols):
                if x == 0:
                    current_average = df.iloc[x, col]
                    average[x,col] = round(current_average,5)
                else:    
                    current_average = np.sum(df.iloc[0:x+1, col])/(x+1)
                    average[x,col] = round(current_average,5)
        return average                
    def average_at_T(T, df):
        entry = Moving_average.average_to_T(T,df)
        return entry[T-1]
    
class Correlation:
    def correlation_toy1y2(df):
        corr = np.zeros([df.shape[1], 2])
        for x in range(df.shape[1]):
            corr_y1 = df.iloc[:,x].corr(df['Y1'])
            corr_y2 = df.iloc[:,x].corr(df['Y2'])
            corr[x][0] = corr_y1
            corr[x][1] = corr_y2
        return corr
    def correlation_average(df):
        corr = Moving_average.average_at_T(df.shape[0],df)
        return Correlation.correlation_matrix(corr) 
    def correleation(df):
        return df.corr()
corr = Correlation.correlation_toy1y2(train_data)

class Test_build:
    def testing_y1(data,corr):
        time,variable = data.shape
        amount = []
        for x in range(time):
            sum = 0
            for var in range(variable):
                i=0
                sum += data.iloc[x,var]*corr[i][0]
                i+=1
            amount.append(sum)
        data.insert(14, 'Test', amount)
        return data 
    def r_value(data,target,corr):
        df = Test_build.testing_y1(data,corr)
        y1_bar = np.mean(target['Y1'])
        train_bar = np.mean(df['Test'])
        time,variable = data.shape
        sum_y1 = 0
        sum_train = 0
        for x in range(time):
            sum_y1 += (target.iloc[x,0] - y1_bar)**2
            sum_train += (df.iloc[x,14]-train_bar)**2
        return 1 - sum_train/sum_y1
    def testing_y2(data,corr):
        time,variable = data.shape
        amount = []
        for x in range(time):
            sum = 0
            for var in range(1,variable):
                i=0
                sum += data.iloc[x,var]*corr[i][1]
                i+=1
            amount.append(sum)
        data.insert(14, 'Test_Y2', amount)
        return data 
    def r_value(data,target,corr):
        df = Test_build.testing_y2(data,corr)
        y1_bar = np.mean(target['Y2'])
        train_bar = np.mean(df['Test_Y2'])
        time,variable = data.shape
        sum_y1 = 0
        sum_train = 0
        for x in range(time):
            sum_y1 += (target.iloc[x,0] - y1_bar)**2
            sum_train += (df.iloc[x,14]-train_bar)**2
        return 1 - sum_train/sum_y1
    def generate(data,corr):
        time,variables = data.shape
        y1_y2 = np.zeros([time,2])
        for x in range(time):
            value_y1 = 0
            value_y2 = 0
            for var in range(2,variables):
                value_y1 += corr[var-2][0]*data.iloc[x,var]
                value_y2 += corr[var-2][1]*data.iloc[x,var]
            y1_y2[x] = [value_y1,value_y2]
        return pd.DataFrame(y1_y2).to_csv('stanley/data/preds.csv')
         
#print(test_data.shape)            
#print(Test_build.r_value(train_var_data,train_target_val,corr))
#Test_build.generate(test_data,corr)

class Fitting_y1:
    def mean_std(df,variables):
        mu_sigma = np.zeros([14,2])
        for val in range(len(variables)):
            mu_sigma[val] = [np.mean(df.iloc[:,val]),np.std(df.iloc[:,val])]       
        return mu_sigma
     
    def log_normal(x, mu, sigma):
        math.log(math.exp(-0.5*((x-mu)/sigma)^2)/(math.sqrt(2*math.pi*sigma^2)), math.e)
    
    def scatters(train_data, test_data, variables):
        max_min_mean_vol_train = np.zeros([len(variables),3])
        max_min_mean_vol_test = np.zeros([len(variables),3])
        for var in range(len(variables)):
            plt.scatter(train_data['time'],train_data.iloc[:,var],alpha=0.2)
            plt.scatter(test_data['time'],test_data.iloc[:,var],alpha=0.2)
            plt.show()        
            max_min_mean_vol_train[var] = [train_data.iloc[:,var].max(),train_data.iloc[:,var].min(),train_data.iloc[:,var].mean(),train_data.iloc[:,var].std()]
            max_min_mean_vol_test[var] = [test_data.iloc[:,var].max(),test_data.iloc[:,var].min(),test_data.iloc[:,var].mean(),test_data.iloc[:,var].std()]
        return max_min_mean_vol_test, max_min_mean_vol_train

print(Correlation.correlation_toy1y2(train_data))