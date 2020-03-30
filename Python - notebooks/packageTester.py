#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:54:17 2020

@author: snehgajiwala
"""

import pandas as pd
#import numpy as np
from TSForecasting.TsForecasting import TimeSeriesForecast


def trainingData():
    """
    Reading Train Data 
    input:
        none
    output:
        data: training dataframe with index => DisplayTime value => GlucoseValues
    """
    
    data = pd.read_csv("~/Desktop/NCSA_genomics/Data/data_hall.txt", sep="\t") #use your path
    #data.head()
    data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data['time_gap'] = data['Display Time']- data['Display Time'].shift(1)
    meta = pd.read_csv("~/Desktop/NCSA_genomics/Data/Hall_meta.csv") 
    for subjectId, df in data.groupby('subjectId'):
        print("==============================================================")
        print("Subject ID: "+str(subjectId))
        temp = meta[meta["ID"]==subjectId]
        print("Status: "+str(temp["status"].values[0]))
        #print(df)
        #print(df['GlucoseValue'].describe())
        #100*(len(df[df["time_gap"]>str("00:05:00")])/df['GlucoseValue'].count())
        print("Length of the readings: "+str(df['GlucoseValue'].count()))
        print("Max. Glucose value: "+str(df['GlucoseValue'].max()))
        print("Min. Glucose value: "+str(df['GlucoseValue'].min()))
        print("Mean Glucose value: "+str(round(df['GlucoseValue'].mean(),3)))
        print("Missing Values: "+str(len(df[df["time_gap"]>str("00:05:00")])))
        print("Percent of missing values: "+str(round(100*(len(df[df["time_gap"]>str("00:05:00")])/df['GlucoseValue'].count()),2))+"%")
        #print(df['DisplayTime'])
        print()
        print("Days: "+str(df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]))
    
    #dropping columns we don't need
    
    
    data.drop(['subjectId', 'Internal Time', 'time_gap'], axis=1, inplace=True)
    
    #Converting the Display Time to 'datetime' so that it can be used as an index
    #data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data = data.set_index(['Display Time'], drop=True)
    #data.head()
    
    return data

def testingData():
    """
    Reading Test Data 
    input:
        none
    output:
        data: testing dataframe with index => DisplayTime value => GlucoseValues
    """
    #reading datasets for training
    data = pd.read_csv("~/Desktop/NCSA_genomics/Data/CGManalyzer.csv")

    diabetic_1 = data[data['ID'] == "ID01"]
    diabetic_2 = data[data['ID'] == "ID11"]
    prediabetic = data[data['ID'] == "ID21"]
    nondiabetic = data[data['ID'] == "ID29"]
    
#     diabetic_1 = pd.read_csv("/Users/snehgajiwala/Desktop/NCSA_genomics/Data/CGManalyzer-datasets/test/ID01.csv")
#     diabetic_2 = pd.read_csv("/Users/snehgajiwala/Desktop/NCSA_genomics/Data/CGManalyzer-datasets/test/ID11.csv")
#     prediabetic = pd.read_csv("/Users/snehgajiwala/Desktop/NCSA_genomics/Data/CGManalyzer-datasets/test/ID21.csv")
#     nondiabetic = pd.read_csv("/Users/snehgajiwala/Desktop/NCSA_genomics/Data/CGManalyzer-datasets/test/ID29.csv")
    
    #Converting the Display Time to 'timeStamp' so that it can be used as an index
    diabetic_1['timeStamp'] = diabetic_1['timeStamp'].apply(lambda x: pd.datetime.strptime(x, '%Y:%m:%d:%H:%M'))
    diabetic_1.drop(['ID'], axis=1, inplace=True)
    diabetic_1 = diabetic_1.set_index(['timeStamp'], drop=True)
    
    prediabetic['timeStamp'] = prediabetic['timeStamp'].apply(lambda x: pd.datetime.strptime(x, '%Y:%m:%d:%H:%M'))
    prediabetic.drop(['ID'], axis=1, inplace=True)
    prediabetic = prediabetic.set_index(['timeStamp'], drop=True)
    
    nondiabetic['timeStamp'] = nondiabetic['timeStamp'].apply(lambda x: pd.datetime.strptime(x, '%Y:%m:%d:%H:%M'))
    nondiabetic.drop(['ID'], axis=1, inplace=True)
    nondiabetic = nondiabetic.set_index(['timeStamp'], drop=True)
    
    diabetic_2['timeStamp'] = diabetic_2['timeStamp'].apply(lambda x: pd.datetime.strptime(x, '%Y:%m:%d:%H:%M'))
    diabetic_2.drop(['ID'], axis=1, inplace=True)
    diabetic_2 = diabetic_2.set_index(['timeStamp'], drop=True)
    
    return diabetic_1, diabetic_2, prediabetic, nondiabetic


train_set = trainingData()
test_set = testingData()


#obj = TimeSeriesForecast()
#obj.connectivityTester()

lstmModel = obj.trainModel(train_set)

diabetic_1, diabetic_2 , prediabetic, non_diabetic = testingData()#these time series' data will be used to plot comparison graphs
diabetic_1_faulty, diabetic_2_faulty, prediabetic_faulty, non_diabetic_faulty = testingData()#gaps will be introduced in these time series' for imputations 

start, end = obj.createGap(diabetic_1)
gap_tester_diabetic_1 = diabetic_1.iloc[start:end+2]
diabetic_1_faulty = obj.faultyData(diabetic_1_faulty,start,end+1)

#here, we're actually running the model and getting the imputed values for the gap
predicted, true = obj.testModel(lstmModel,gap_tester_diabetic_1)
#here we are are filling in the gap we created with imputed values generated by the model
for i in range(0,501):
     diabetic_1_faulty['glucoseValue'][start+i] = predicted[i][0]

obj.plot(diabetic_1_faulty['glucoseValue'].tolist(),diabetic_1['glucoseValue'].tolist())
obj.getMetrics(predicted,true)


#repeating the same for diabetic type 2
start, end = obj.createGap(diabetic_2)
gap_tester_diabetic_2 = diabetic_2.iloc[start:end+2]
diabetic_2_faulty = obj.faultyData(diabetic_2_faulty,start,end+1)

predicted, true = obj.testModel(lstmModel,gap_tester_diabetic_2)

for i in range(0,501):
    diabetic_2_faulty['glucoseValue'][start+i] = predicted[i][0]
    
obj.plot(diabetic_2_faulty['glucoseValue'].tolist(),diabetic_2['glucoseValue'].tolist())
obj.getMetrics(predicted,true)



#repeating the same for prediabetic
start, end = obj.createGap(prediabetic)
gap_tester_prediabetic = prediabetic.iloc[start:end+2]
prediabetic_faulty = obj.faultyData(prediabetic_faulty,start,end+1)

predicted, true = obj.testModel(lstmModel,gap_tester_prediabetic)

for i in range(0,501):
    prediabetic_faulty['glucoseValue'][start+i] = predicted[i][0]
    
obj.plot(prediabetic_faulty['glucoseValue'].tolist(),prediabetic['glucoseValue'].tolist())
obj.getMetrics(predicted,true)



#repeating the same for non-diabetic
start, end = obj.createGap(non_diabetic)
gap_tester_non_diabetic = non_diabetic.iloc[start:end+2]
non_diabetic_faulty = obj.faultyData(non_diabetic_faulty,start,end+1)

predicted, true = obj.testModel(lstmModel,gap_tester_non_diabetic)

for i in range(0,501):
    non_diabetic_faulty['glucoseValue'][start+i] = predicted[i][0]
    
obj.plot(non_diabetic_faulty['glucoseValue'].tolist(),non_diabetic['glucoseValue'].tolist())
obj.getMetrics(predicted,true)

print("End")
