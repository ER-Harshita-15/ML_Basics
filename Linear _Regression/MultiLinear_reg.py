import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
print('Libraries are imported successfully')

#Creating Dataset
df=pd.DataFrame({'area':[2600,3000,3200,3600,4000],'bedrooms':[3,4,np.nan,3,5],'age':[20,15,18,30,8],'price':[550000,565000,610000,595000,760000]})
#print(df)

#DATA PREPROCESSING
#To fill Nan value : Doing Calculations Required
import math 
median_bedrooms=math.floor(df.bedrooms.median())
#print(median_bedrooms)

#Filling the Nan Value

df.bedrooms=df.bedrooms.fillna(median_bedrooms)

print(df)

#Model creation
reg=linear_model.LinearRegression()  #object of model is created
reg.fit(df[['area','bedrooms','age']],df.price)

m=reg.coef_
print(m)
b=reg.intercept_
print(b)

print(reg.predict([[3000,3,40]]))

#manual
Price=m[0]*3000 +m[1]*3+m[2]*40+b
print(Price)

print(reg.predict([[2500,4,5]]))

