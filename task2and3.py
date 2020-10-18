# coding: utf-8
from os import system, name   
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from warnings import simplefilter
import pyfiglet 

#suppress FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

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
    print("Task 2 and 3 | Assignment 1 | CSOE 17 - Big Data Analysis | Roll no : 114117098")
    input("...| Press any key to Begin |...")
    print("")

#function to structure the data
def structure_data(data,t):
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
    
    if t:
        data["quality"] = data["quality"].apply(lambda x: "Good" if(x>=7) else "Bad")
    y=data["quality"]
    data=data.drop("quality",axis=1)    
    data=pd.DataFrame(StandardScaler().fit_transform(data),columns=data.columns)
    data["quality"]=y
    return data

#function to predict
def mod_predict(X_test, y_test, mod,l):
    
    y_pred = mod.predict(X_test)
    if l:
        for i in range(len(y_pred)):
            if y_pred[i]>=7:
                y_pred[i]=1
            else:
                y_pred[i]=0
        for i in range(len(y_test)):
            if y_test[i]>=7:
                y_test[i]=1
            else:
                y_test[i]=0
            
        
    
    
    accuracy = accuracy_score(y_test, y_pred)
    
    con_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(con_mat)
    print("")
    precision = con_mat[0,0]/(con_mat[0,0]+con_mat[0,1])
    recall = con_mat[0,0]/(con_mat[0,0]+con_mat[1,0])
    f1 = (2*precision*recall)/(precision+recall)
    sensitivity = con_mat[0,0]/(con_mat[0,0]+con_mat[0,1])
    specificity = con_mat[1,1]/(con_mat[1,0]+con_mat[1,1])
    print("The Accuracy, Precision, Recall, F1 score, Sensitivity and specificity are: ")
    print(accuracy, precision, recall, f1, sensitivity, specificity)

#reading the data
train_data=pd.read_csv("wineQualityRed_train.csv",header=None)
test_data=pd.read_csv("wineQualityRed_test.csv", header=None)

#structuring the data
train= structure_data(train_data,True)
test= structure_data(test_data,True)

#Classifiers

# Logistic Regression
mod_log = LogisticRegression(max_iter=100000)
mod_log.fit(train.drop("quality", axis = 1), train["quality"])

# SVM Classifier
mod_svc = SVC( kernel = 'linear')
mod_svc.fit(train.drop("quality", axis = 1), train["quality"])

#Naive Bayes
mod_nav_bay = GaussianNB()
mod_nav_bay.fit(train.drop("quality", axis = 1), train["quality"])

#Linear Regression as a classifier
train_l=structure_data(train_data, False)
test_l=structure_data(test_data, False)
mod_lin= LinearRegression()
mod_lin.fit(train_l.drop("quality", axis = 1), train_l["quality"])

# 3...2...1... Go!



input("Press enter for Logistic Regression")
print("Logistic Regression")
mod_predict(test.drop("quality", axis = 1), test["quality"], mod_log, False)
print("\n")

input("Press enter for Linear Regression")
print("Linear Regression")
mod_predict(test_l.drop("quality", axis = 1), test_l["quality"], mod_lin, True)
print("\n")

input("Press enter for SVM")
print("SVM")
mod_predict(test.drop("quality", axis = 1), test["quality"], mod_svc, False)
print("\n")

input("Press enter for Naive Bayes")
print("Naive Bayes")
mod_predict(test.drop("quality", axis = 1), test["quality"], mod_nav_bay, False)
print("\n")
result = pyfiglet.figlet_format("Thank you!") 
print(result)
