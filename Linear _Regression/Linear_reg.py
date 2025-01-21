import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model

#Creating Dataset
data= { 'Area':[2600,3000,3200,3600,4000],'Price':[550000,565000,610000,680000,725000]}
df = pd.DataFrame(data)
print(df)

# Plotting the data points
plt.xlabel('Area(sq ft)')
plt.ylabel('Price(US $)')
plt.scatter(df.Area,df.Price,color='red',marker='+')
#plt.show()

#Defining the Model
reg = linear_model.LinearRegression()
reg.fit(df[['Area']],df.Price)

#Predicting Variable 
prediction=reg.predict(pd.DataFrame({'Area':[3300]}))

#Making predictions
print(f"The Price of house with 3300 sq. ft. area is ${prediction[0]}")

#MANNUAL CALCULATIONS 

# Slope(m) [change in y for per unit change in x] 
m=reg.coef_
print(f"Slope of regression line : {m[0]}")

#intercept(b) [value of y for x=0]
b=reg.intercept_
print(f"The intercept at Y-axis : {b}")

#manual prediction [y=mx+b]
prediction_manual= m*3300 + b 
print(f"The cost of house with 3300 sq. ft. area is {prediction_manual[0]}")


#Performing More Operations 
d=pd.read_excel("house_area.xlsx")
p=reg.predict(d[["Area"]])
new_data=pd.DataFrame({"Area":d['Area'],'Predicted_Price':p})

#new_data.to_excel("House_Predicted_Price.xlsx",index=False)

#PLOTTING LINEAR REGRESSION'S LINE GRAPH
plt.xlabel('Area',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.scatter(df.Area,df.Price,color='red',marker='+')
plt.plot(df.Area,reg.predict(df[['Area']]),color='blue')
plt.show()