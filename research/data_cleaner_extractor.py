import pandas as pd
import numpy as np
import math
import matplotlib as mp


data = pd.read_csv('test.csv')
df = pd.DataFrame(data)
print(df.head())

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
                    average[x,col] = current_average
                else:    
                    current_average = np.sum(df.iloc[0:x+1, col])/(x+1)
                    average[x,col] = current_average
        return average                
    def average_at_T(T):
        entry = Moving_average.average_to_T(T,df)
        return entry[T]

Moving_average.average_to_T(5, df)