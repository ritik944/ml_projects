#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"C:\Users\ritik\OneDrive\Documents\GitHub\jetson_nano_projects\house_price_pred\california_housing_train.csv")
df_test=pd.read_csv(r"C:\Users\ritik\OneDrive\Documents\GitHub\jetson_nano_projects\house_price_pred\california_housing_test.csv")

#%%
# print(df.head())

# print(df.columns)

# print(df.info())

# print(df.describe())

#%%
plt.figure(figsize=(10,6))
coordintes_vs_population=plt.scatter(x=df['longitude'].values,y=df['latitude'].values,alpha=.1,s=df['population'].values/50,label='population', c=df['median_house_value'].values,cmap=plt.get_cmap("jet"),)
plt.colorbar(coordintes_vs_population)
# plt.show()


#%%
from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age","population"]
scatter_matrix(df[attributes],figsize=(12,8))

#%%
# correlation between median_house_value and median_income is much more corelated then others
correlation=df.corr()
print(correlation['median_house_value'])
# %% data prep
x_train=df.iloc[:,:8].values
y_train=df.iloc[:,-1].values

scaler =StandardScaler()
x_train=scaler.fit_transform(x_train)

x_test=df_test.iloc[:,:8].values
y_test=df_test.iloc[:,-1].values

x_test=scaler.transform(x_test)
print(x_train.shape,y_train.shape)
print(x_test.shape ,y_test.shape)

#%% model

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)
y_test_pred=model.predict(x_test)
print(y_test.shape,y_test_pred.shape)

#%%
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse_loss=mean_squared_error(y_test,y_test_pred)
mae_loss=mean_absolute_error(y_test,y_test_pred)

print(mae_loss,mse_loss)
print(round(model.score(x_train,y_train),4)*100,'%')
print(round(model.score(x_test,y_test),4)*100,'%')
#direct comparision
print(y_test[0],y_test_pred[0])

print(np.argmin(y_test_pred))
print(y_test[2186],y_test_pred[2186])