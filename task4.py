# coding: utf-8 
# Some interactive code commented out to comply with the deliverable instructions. 
from os import system, name  
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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
    print("Task 4 | Assignment 1 | CSOE 17 - Big Data Analysis | Roll no : 114117098")
    #input("...| Press any key to Begin |...")
    print("")

def structure_data(data):

    data.columns = ["Col_1"]
    data = data.Col_1.str.split(";", expand = True)    
    col = []
    for i in data.iloc[0]:
        col.append(i.strip('\"'))    
    data.columns = col
    data = data[1:]    
    for i in range(12):
        data[data.columns[i]] = data[data.columns[i]].astype(float)    
    data = data.reset_index(drop = True)    
    data["quality"] = data["quality"].apply(lambda x: 1 if(x>=7) else 0)    
    StandardScaler().fit_transform(data.drop("quality", axis = 1))    
    return data

def logistic_model(X_train, y_train):    
    
    mod = LogisticRegression(max_iter=100000)    
    mod.fit(X_train, y_train)    
    return mod

def svm_classifier(X_train, y_train):    
    
    mod = SVC(kernel='linear')    
    mod.fit(X_train, y_train)    
    return mod

def naive_bayes(X_train, y_train):    
    
    mod = GaussianNB()    
    mod.fit(X_train, y_train)    
    return mod

def mod_predict(X_test, y_test, mod):
    
    y_pred = mod.predict(X_test)    
    accuracy = accuracy_score(y_test, y_pred)    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("")
    precision = cm[0,0]/(cm[0,0]+cm[0,1])
    recall = cm[0,0]/(cm[0,0]+cm[1,0])
    f1 = (2*precision*recall)/(precision+recall)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    print("The Accuracy, Precision, Recall, F1 score, Sensitivity and specificity are: ")
    print(accuracy, precision, recall, f1, sensitivity, specificity)

def pca_transform(X, n):    
    
    pca = PCA(n_components=n)    
    X = pca.fit_transform(X)    
    return X

def print_results(X_train, X_test, y_train, y_test, n):

    print("Number of attributes: " + str(n))
    print("\n")    
    
    X_train_n = pca_transform(X_train, n)
    X_test_n = pca_transform(X_test, n)    
    logistic_regression = logistic_model(X_train_n, y_train)
    svm_model = svm_classifier(X_train_n, y_train)
    nb_model = naive_bayes(X_train_n, y_train) 
    
    print("Logistic Regression")
    mod_predict(X_test_n, y_test, logistic_regression)
    print("\n")

    print("SVM")
    mod_predict(X_test_n, y_test, svm_model)
    print("\n")

    print("Naive Bayes")
    mod_predict(X_test_n, y_test, nb_model)
    print("\n")

#reading the data
train_data = pd.read_csv("wineQualityRed_train.csv", header=None)
test_data = pd.read_csv("wineQualityRed_test.csv", header=None)

#structuring the data
train_data_prep = structure_data(train_data)
test_data_prep = structure_data(test_data)

#train the data
X_train = train_data_prep.drop("quality", axis = 1)
y_train = train_data_prep["quality"]

X_test = test_data_prep.drop("quality", axis = 1)
y_test = test_data_prep["quality"]

print_results(X_train, X_test, y_train, y_test, 7)
print_results(X_train, X_test, y_train, y_test, 4)

