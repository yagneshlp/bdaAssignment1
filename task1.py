# coding: utf-8
from os import system, name   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#function to clear the screen 
def clear():
    # for windows 
    if name == 'nt': 
        _ = system('cls')
    # for mac and linux
    else: 
        _ = system('clear') 

clear()

if __name__ == '__main__':
    print("Task 1 | Assignment 1 | CSOE 17 - Big Data Analysis | Roll no : 114117098")
    input("...| Press any key to Begin |...")
    print("")

#function to structure the data
def structure_data(data):
    data.columns=["col1"]
    data=data.col1.str.split(";", expand= True)
    col=[]
    for i in data.iloc[0]:
        col.append(i.strip('\"'))
    data.columns=col
    data=data[1:]
    data = data.reset_index(drop = True)
    for i in range(12):
        data[data.columns[i]] = data[data.columns[i]].astype(float)
    y=data["quality"]    
    data=pd.DataFrame(StandardScaler().fit_transform(data.drop("quality",axis=1)),columns=data.drop("quality",axis=1).columns)
    data["quality"]=y    
    return data

#reading the data
train_data=pd.read_csv("wineQualityRed_train.csv",header=None)
test_data=pd.read_csv("wineQualityRed_test.csv", header=None)

#structuring the data
train = structure_data(train_data)
test = structure_data(test_data)

#fitting into LR
lr = LinearRegression()
model=lr.fit(train.drop("quality",axis=1),train["quality"])

#testing data
pred=model.predict(test.drop("quality",axis=1))
meanSqErr=mean_squared_error(test["quality"],pred)
sumSqErr=meanSqErr*len(test.index)

print("•••||| Sum of square error: " + str(sumSqErr) + " |||•••" )
print("\nPreparing Plot...")
#lm plot
for i in range(12):
    sns.lmplot(y= test.columns[i], x= "quality", data = test)

input("\n ♦ Plot Prepared! Press any key to show. ")   
plt.show() 






