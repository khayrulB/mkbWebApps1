# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:09:02 2022
Diabetes CLASSIFICATION
@author: khayr
"""

#------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import datetime
import itertools
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from collections import Counter

#--------------------------------------------------------
# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#-------------------------------------------------------

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
#from imblearn.ensemble import BalancedRandomForestClassifier

import string
import re
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score, confusion_matrix, auc, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

import spacy
#import scipy.stats as stats
from tqdm import tqdm    #, tqdm_notebook, tnrange
#-----------------------------------
#import getPlots as kplt
#import corona_sentiment_functions as csf
#import optimize_modelPara as omp
#from nrclex import NRCLex

tqdm.pandas(desc='Progress')
#print("\n I am KKK and now here")



def get_trained_model(Xtrn, yTrn, Xval, yVal, predModel):
    # -------------------------------------------------------------------
    # Word2vector from GENSIM
    # https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
    # -------------------------------------------------------------------

    # fit() & predict() methods : accepts a 2d array, e.g., numpy ndarray as input
    # fit() can accept X_train as ndarray and y_train as a 1d list
    if predModel == 'LR': # Linear Regression
        clf = LinearRegression(random_state=42)
        clf.fit(Xtrn, yTrn)
        yPred = clf.predict(Xval)
        yPred =[1 if x >=0.7 else 0 for x in yPred]
        print('LINREG countVect accuracy %s' % accuracy_score(yVal, yPred))
        print('LINREG countvect Weighted F1_score %f ' % (f1_score(yVal, yPred, average='weighted')))
        
    elif predModel == 'LOGREG':
        
        #omp.get_best_hyperPara_for_LR(Xtrn, yTrn, Xval, yVal)
        #-------LOGREG  with countVect-----------------------------------------
        #optimal: 0.838269, 'C':1.0, 'penalty':'l2', solver:'liblinear'
        clf = LogisticRegression(C=1.0, solver='liblinear', penalty='l2', max_iter=500,  multi_class='auto', random_state=42)
        #clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200,  multi_class='auto', random_state=42)
        clf.fit(Xtrn, yTrn)
        yPred = clf.predict(Xval)
        print('LOGR countVect Precision %s' % precision_score(yVal, yPred))
        print('LOGR countVect Recall %s' % recall_score(yVal, yPred))
        print('LOGREG countVect accuracy %s' % accuracy_score(yVal, yPred))
        print('LOGREG countvect Weighted F1_score %f ' % (f1_score(yVal, yPred, average='weighted')))
        
    elif predModel =='RF':
        # ---RF with Count-vector ---------------------------------------------
        # https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        
        # omp.get_best_hyperPara_for_RF(Xtrn, yTrn, Xval, yVal)
        # Best: 0.866200 using max_features:'log2', n_estimators:1000
            
        #rf1 = BalancedRandomForestClassifier(n_estimators=800, min_samples_split=10, min_samples_leaf= 2, max_features ='sqrt', max_depth=30, bootstrap=False, random_state=42)
        clf = RandomForestClassifier(n_estimators=500, random_state=42)
        clf.fit(Xtrn, yTrn) # return trained classifier object
        yPred = clf.predict(Xval) # convert DF to ndarray
        print(yPred)
        print('RF countVect Precision %s' % precision_score(yVal, yPred))
        print('RF countVect Recall %s' % recall_score(yVal, yPred))
        print('RF countVect accuracy %s' % accuracy_score(yVal, yPred))
        print('RF Weighted F1_score %f ' % (f1_score(yVal, yPred, average='weighted')))#weighted for imbala. data    
        #Acc=83.68, F1=80.54 for binary
        
    elif predModel == 'LSVM':
        
        #omp.get_best_hyperPara_for_SGDClassifier(Xtrn, yTrn, Xval, yVal)
        #--------------------------------------------------------------------------
        #clf = SVC(C=10, gamma=0.01, kernel='sigmoid', random_state=42) # NoN-Linear
        clf = LinearSVC(max_iter=20000, random_state=42, tol=1e-5) # Linear
        #clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42) # Like LinearSVC, but more memory efficient
        clf.fit(Xtrn, yTrn)
        yPred = clf.predict(Xval)
        print('LSVM Precision %s' % precision_score(yVal, yPred))
        print('LSVM Recall %s' % recall_score(yVal, yPred))
        print('LSVM accuracy %s ' % (accuracy_score(yVal, yPred)))
        print('LSVM Weighted F1_score %f ' % (f1_score(yVal, yPred, average='weighted')))
        #Acc=82.77, F1=80.34
    elif predModel == 'NB':
        
        # omp.get_best_hyperPara_for_NB(Xtrn, yTrn, Xval, yVal)
        # Best: 0.819934 using {'var_smoothing': 0.0657933224657568}
        #--------------------------------------------------------------------------
        clf = GaussianNB(var_smoothing = 0.1) #0.003511 # random_state=42 is not a parameter in NB
        clf.fit(Xtrn, yTrn)
        yPred = clf.predict(Xval)
        print (type(clf))
        #print(nb)
        print('NB countVect Precision %s' % precision_score(yVal, yPred))
        print('NB countVect Recall %s' % recall_score(yVal, yPred))
        print('NB accuracy %s ' % (accuracy_score(yVal, yPred)))
        print('NB Weighted F1_score %f' % (f1_score(yVal, yPred, average='weighted')))
        #Acc=82.62, F1=80.86, var_smoothing=0.1
        
    elif predModel == 'DL':
        # define the keras model
        clf = Sequential()
        clf.add(Dense(24, input_shape=(Xtrn.shape[1],), activation='relu')) # No of cols=No of vars
        clf.add(Dense(12, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("\n model compilation is over")

        print("\n Training is going on")

        # Train or fit the keras model on the dataset
        clf.fit(Xtrn, np.array(yTrn), epochs=120, batch_size=10)
        # evaluate the keras model
        _, accuracy = clf.evaluate(Xtrn, np.array(yTrn))

        print ("\n Training Accuracy: ", accuracy)
        print("\n Training is over")
        # make class predictions with the model
        yPred = (clf.predict(Xval) > 0.5).astype(int)

        print('DL countVect Precision %s' % precision_score(yVal, yPred))
        print('DL countVect Recall %s' % recall_score(yVal, yPred))
        print('DL accuracy %s ' % (accuracy_score(yVal, yPred)))
        print('DL Weighted F1_score %f' % (f1_score(yVal, yPred, average='weighted')))


        
    elif predModel == 'HYBRID':
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500,  multi_class='auto', random_state=42)
        clf.fit(Xtrn, yTrn)
        yPred1 = clf.predict(Xval)
        #-------------------------------------
        clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42) # Like LinearSVC, but more memory efficient
        clf.fit(Xtrn, yTrn)
        yPred2 = clf.predict(Xval)
        
        yPred=[]
        for i in range(len(yPred1)):
            if (yPred1[i]==1) or (yPred2[i]==1):
                yPred.append(1)
            else:
                yPred.append(0)
        print('HYBRID countVect Precision %s' % precision_score(yVal, yPred))
        print('HYBRID countVect Recall %s' % recall_score(yVal, yPred))
        print('HYBRID countVect accuracy %s' % accuracy_score(yVal, yPred))
        print('HYBRID countvect Weighted F1_score %f ' % (f1_score(yVal, yPred, average='weighted')))
         
    else:
        print ("\n Enter Correct prediction Model?")
        print("exiting the program")
        print(sys.exit())
        
    
    return clf 
#----------------------------------------------------------------------------

#==============================================================================
# https://www.youtube.com/watch?v=xl0N7tHiwlw 
#=============================================================================
if __name__ =="__main__":
    
    
    predModel= "LOGREG" # DL/LSVM/NB/RF/LOGREG/


    #dataDir = "D:/TKFD/Projects/mentalHealth/depression/data/"
    #dataDir = "D:/TKFD/Projects/mentalHealth/data2022/dataSets/"
    dataDir = "D:/TKFD/Projects/mlApps/DiabetesDR/"

    df = pd.read_csv(dataDir + "pima-indians-diabetes.csv") #two-cols: [tweet, target]
    #print(df.head())
    #dataset = loadtxt(dataDir +'pima-indians-diabetes.data.txt', delimiter=',')

    
    #Remove NaN rows in Y column
    df=df[df['Y'].notnull()]
    #print(df.head())
    #print(df.info())

    # Input Variables (X):

    #1 Number of times pregnant [PN]
    #2 Plasma glucose concentration at 2 hours in an oral glucose tolerance test [PGC]
    #3 Diastolic blood pressure (mm Hg) [DBP]
    #4 Triceps skin fold thickness (mm) [TrST]
    #5 2-hour serum insulin (mu U/ml) [SI]
    #6 Body mass index (weight in kg/(height in m)^2) [BMI]
    #7 Diabetes pedigree function [DPF]
    #8 Age (years) [AGE]
    #Output Variables (y):
    # Class variable (0 or 1)--> 1 diabetes  

    # split into input (X) and output (y) variables
    dataset =df.values #  creates a 2D array
    #print(dataset[:5])
    cols=['PN','PGC', 'DBP','TrST', 'SI', 'BMI', 'DPF', 'AGE']
    X = df[cols].values # 0 to 7 
    y = df['Y'].values 
    yy=dataset[:,8]
    # print (yy[:10])
    # print (X[:10])
    # print(y[:10])
    # -----------------------------------------------------------------
    # Split Numerical 2D feaArray into training and testing sets
    #------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.3, shuffle=True)
    
    #-------------------------------------------------------
    # GET Trained Model (validation for train set is inside)
    #--------------------------------------------------------
    trained_model = get_trained_model(X_train, y_train, X_val, y_val, predModel)
    # print("\n DataType for X_train & y_train(label)")
    # print(type(X_train))  # --> numpy.ndarray
    # print(type(y_train))  # --> list
    
    #Save TRAINED MODEL
    
    import pickle
    data={'model': trained_model}
    with open('saved_trained_' + predModel + '.pkl', 'wb') as file: 
        pickle.dump(data, file)

    # REtrive saved data from pickle file
    with open('saved_trained_' + predModel + '.pkl', 'rb') as file: 
        data = pickle.load(file)
        
    model_loaded= data['model']
    
    
    
    y_pred = model_loaded.predict(X_val)
    #Predicted from the SAVED model 
    #print(y_pred[:10])

    


    x1={'PN':6., 'PGC':148., 'DBP':78., 'TrST':50., 'SI':525., 'BMI':33.6,'DPF': 1.5,'AGE':50.0}
    xNew = list(x1.values())
    
    y1_pred = (model_loaded.predict(np.array([xNew])) > 0.5).astype(int) # Make 2D array
    
    print("\n Prediction of patient X1 is ", y1_pred)
#=============================================================================
    import streamlit as st
    import pickle 
    import numpy as np

    def show_predict_page():
        st.title("Software for Diabetes Detection: SoftDDR")
        name=st.sidebar.text_input('Enter patient name')
        if not name:
            st.sidebar.warning("Please fill out so required fields")
        st.write ("""### We need some info to predictict diabetic status""")
    
        a = st.slider("PN", 0.0, 17.0,0.0) # min, max, start
        b = st.slider("PGC", 0.0, 199.0,0.0)
        c = st.slider("DBP", 0.0, 122.0, 0.0)
        d = st.slider("TrST", 0.0, 99.0, 0.0)
        e = st.slider("SI", 0.0, 846.0, 0.0)
        f = st.slider("BMI", 0.0, 67.1, 0.0)
        g = st.slider("DPF", 0.0, 2.42, 0.078)
        h = st.slider("AGE", 0.0, 81.0, 21.0)
    
        ok=st.button("Predict Diabetes Status (Yes or No")
        if ok:
            x1={'PN':a,'PGC':b, 'DBP':c,'TrST':d, 'SI':e, 'BMI':f, 'DPF':g, 'AGE':h}
            #x1={'PN':a, 'PGC':b, 'BMI':c,'AGE':d}
            xNew = list(x1.values())
            #st.subheader(f" variables : {xNew}")
            pred_class= (model_saved.predict(np.array([xNew])) > 0.5).astype(int) # Make 2D array
            #st.subheader(f" pred_class : {pred_class}")
            if pred_class == [0]:
                status='non-diabetic'
            else:
                status ='diabetic'
            st.subheader(f"The patient {name} has : {status}")
      
       
    
#=============================================================================
    # Function to load the model 

    def load_model():
        with open('saved_trained_LOGREG.pkl', 'rb') as file: 
            data = pickle.load(file)
        
        return data 

    # Function call.
    data =load_model()
    # Execute Function
    model_saved= data['model']
#=============================================================================

    #from predict_page import show_predict_page 

    show_predict_page()