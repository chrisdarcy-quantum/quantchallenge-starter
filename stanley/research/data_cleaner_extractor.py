import pandas as pd
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt


data = pd.read_csv('../data/train.csv')
df = pd.DataFrame(data)

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
    def plot_correlation(df, x, y):
        plt.scatter(df.get[:,x], df.get[:,y])
        plt.show()
Moving_average.plot_correlation(df, 'A', 'B')        