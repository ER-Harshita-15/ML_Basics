import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Creating Dataset
df=pd.DataFrame({"Experience":[np.nan,np.nan,'five','two','seven','three','ten','eleven'],"test_score":[8,8,6,10,9,7,np.nan,7],'Interview_score':[9,6,7,10,6,10,7,8],'Salary($)':[50000,45000,60000,65000,70000,62000,72000,80000]})
print(df)

#DATA PREPROCESSING

#Filling nan:
df["Experience"]=df["Experience"].fillna('zero')
df["test_score"]=df["test_score"].fillna(df['test_score'].mean())
print(df) 

#Converting the Experience Column's values from word to number
from word2number import w2n
df["Experience"]=df['Experience'].apply(w2n.word_to_num)
print(df)

#Model Definition
reg=linear_model.LinearRegression()
reg.fit(df[['Experience','test_score','Interview_score']],df['Salary($)'])

#MANUAL CALCULATIONS 
m=reg.coef_
print(f"The slope of line for linear regression:{m}")
b=reg.intercept_
print(f"The intercept on y-axis is {b}")

predict_1=reg.predict(pd.DataFrame([[2,9,6]],columns=['Experience','test_score','Interview_score']))
predict_2=reg.predict(pd.DataFrame([[12,10,10]],columns=['Experience','test_score','Interview_score']))

import math 
print(f"The Salary of a candidate with 2 years of experience,9 out of 10 test score and 6 out of 10 interview score is ${math.floor(predict_1[0])}")
print(f"The Salary of a candidate with 2 years of experience,9 out of 10 test score and 6 out of 10 interview score is ${math.floor(predict_2[0])}")