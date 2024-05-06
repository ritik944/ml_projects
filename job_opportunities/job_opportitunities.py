#%%
import numpy as np
import pandas as pd

df=pd.read_csv(r'C:\Users\ritik\OneDrive\Documents\GitHub\jetson_nano_projects\job_opportunities\job.csv')

#%%
print(df.head())
print(df.describe())
print(df.info())

#%%
print(df.columns)
