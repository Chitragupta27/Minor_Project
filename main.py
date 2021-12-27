#!/usr/bin/env python
# coding: utf-8

# In[7]:


import scipy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def main():
    st.title("Flood prediction")
    x = pd.read_csv(
        "C:\\Users\\Chitragupta Anubhav\\Documents\\Minor_Project\\kerala.csv")


    y = pd.read_csv(
    "C:\\Users\\Chitragupta Anubhav\\Documents\\Minor_Project\\kerala.csv")

    y1 = list(x["YEAR"])
    x1 = list(x["Jun-Sep"])
    z1 = list(x["JUN"])
    w1 = list(x["MAY"])

    plt.plot(y1, x1, '*')
    plt.show()


    flood = []
    june = []
    sub = []

    # CREATING A NEW COLOUMN WITH BINARY CLASSIFICATION DEPENDING IF THAT YEAR HAD FLOODED OR NOT, USING RAINFALL OF THAT YEAR AS THRESHOLD
    # print(x1[114])
    for i in range(0, len(x1)):
        if x1[i] > 2400:
            flood.append('1')
        else:
            flood.append('0')

    # print(len(x1))

    # APPROAXIMATELY FINDING THE RAINFALL DATA FOR 10 DAYS FOR THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2017
    for k in range(0, len(x1)):
        june.append(z1[k]/3)

    # FINDING THE INCREASE IN RAINFALL FROM THE MONTH OF MAY TO THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2017
    for k in range(0, len(x1)):
        sub.append(abs(w1[k]-z1[k]))


    df = pd.DataFrame({'flood': flood})
    df1 = pd.DataFrame({'per_10_days': june})

    x["flood"] = flood
    x["avgjune"] = june
    x["sub"] = sub

    # SAVING THE NEW CSV FILE WITH THE NEW COLOUMNS
    x.to_csv("out1.csv")
    print((x))


    # TAKING THE COLOUMNS WHICH ARE TO USED FOR TRAINING THE MODEL
    # 16 MAR-MAY
    # 20- AVG OF 10 DAYS JUNE
    # 21- DIFFERENCE OF RAINFALL FROM MAY TO JUNE
    # 19 - BINARY CLASS OF FLOOD- 0 OR 1
    # MORE DATA CAN BE ADDED FOR TRAINING, BY JUST ADDING MORE NUMBER OF COLOUMNS FROM THE CSV FILE

    # WE USE LOGISTIC REGRESSION FOR TRAINING

    X = x.iloc[:, [16, 20, 21]].values
    y1 = x.iloc[:, 19].values

    (X_train, X_test, Y_train, Y_test) = train_test_split(X, y1, random_state=0)


    #X1= scale(X)
    # print(X1)

    Lr = LogisticRegression()

    Lr.fit(X, y1)
    print(Lr.score(X, y1))  # PRINTS THE ACCURACY
    # ypred=Lr.score(X_test,Y_test)
    # print(ypred)


    l = [[50, 300, 205]]

    # print(X)

    # ypred=Lr.predict(X)
    f1 = Lr.predict(l)

    for i in range(len(f1)):

        if (int(f1[i]) == 1):
            print(f1[i], "- possibility of  severe flood")
        else:
            print(f1[i], "- no chance of severe flood")


if __name__ == '__main__':
    main()
