#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:37:04 2020

@author: snehgajiwala
"""

import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
#from keras.optimizers import Adam
from keras.layers import LSTM


from scipy import stats

from tabulate import tabulate
pdtabulate=lambda df:tabulate(df,headers='keys')

import random
import re
from dateutil.parser import parse

import warnings  
warnings.filterwarnings('ignore')

import os


class TimeSeriesForecast:
    lstm_model = None
    
    cwd = os.getcwd()

    def_testing = pd.read_csv(cwd+'/TSForecasting/Data/Hall_test.txt', sep=",")

    def __init__(self):
        """
        The __init__ method initializes and trains the LSTM model on the embedded Hall Dataset
        input:
            None
        output:
            A trained model that can be used for imputations
        """
        data = pd.read_csv(self.cwd+'/TSForecasting/Data/Hall_data.csv') #use your path
        #data.head()
        #data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
     
        #dropping columns we don't need
        
        
        data.drop(['subjectId'], axis=1, inplace=True)
        
        #Converting the Display Time to 'datetime' so that it can be used as an index
        length = data.shape[0]
        length

        # for i in range(0,length):
        #     s = str(data.iloc[i]['Display Time'])
        #     k = re.sub("[^0-9]", "", s)
        #     datetimeObj = parse(k) 
        #     data = data.replace(to_replace = s, value = datetimeObj)

        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data = data.set_index(['Display Time'], drop=True)
        #data.head()
        
        #train = data

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_sc = scaler.fit_transform(data)
    
        #Reshaping the data to work for an LSTM network
    
        train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=data.index)
    
    
        for s in range(1,2):
            train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
    
        X_train = train_sc_df.dropna().drop('Y', axis=1)
        y_train = train_sc_df.dropna().drop('X_1', axis=1)
    
    
        X_train = X_train.as_matrix()
        y_train = y_train.as_matrix()
    
    
        X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
       
    
        #print('Train shape: ', X_train_lmse.shape)
        
    
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(7, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
        self.lstm_model.add(Dense(1))
        self.lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        history_lstm_model = self.lstm_model.fit(X_train_lmse, y_train, epochs=3, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
   
    
    def datePreprocess(self,data):
        """
        The datePreprocess method is used to preprocess the testing data
        it indentifies the date and converts it to the standard datetime format
        it also converts the Timestamp to the index
        input:
            data: dataset we wish to convert the timestamp of
        output:
            data: dataset with the converted timestamp
        """
        # data = data.reset_index()
        length = data.shape[0]
        for i in range(0,length):
            #print(i)
            s = str(data.iloc[i]['Timestamp'])
            k = re.sub("[^0-9]", "", s)
            datetimeObj = parse(k) 
            data = data.replace(to_replace = s, value = datetimeObj)
        
        data = data.set_index(['Timestamp'], drop=True)
        
        return(data)


    def train(self, data):
        """
        The train method is used to train the model on user supplied data
        input:
            data: dataset we want to train the model on
        output:
            A trained model that can be used for imputations
        """
		#use your path
        #data.head()
        #data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
     
        #dropping columns we don't need
        
        
        
        
        #Converting the Display Time to 'datetime' so that it can be used as an index
        # length = data.shape[0]
        #length

        # for i in range(0,length):
        #     s = str(data.iloc[i]['Display Time'])
        #     k = re.sub("[^0-9]", "", s)
        #     datetimeObj = parse(k) 
        #     data = data.replace(to_replace = s, value = datetimeObj)

        # data['Timestamp'] = data['Timestamp'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # data = data.set_index(['Timestamp'], drop=True)
        #data.head()
        
        #train = data

        data = self.datePreprocess(data)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_sc = scaler.fit_transform(data)
    
        #Reshaping the data to work for an LSTM network
    
        train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=data.index)
    
    
        for s in range(1,2):
            train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
    
        X_train = train_sc_df.dropna().drop('Y', axis=1)
        y_train = train_sc_df.dropna().drop('X_1', axis=1)
    
    
        X_train = X_train.as_matrix()
        y_train = y_train.as_matrix()
    
    
        X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
       
    
        print('Train shape: ', X_train_lmse.shape)
        
    
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(7, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
        self.lstm_model.add(Dense(1))
        self.lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        history_lstm_model = self.lstm_model.fit(X_train_lmse, y_train, epochs=3, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])


    def dataDescribe(self):
        """
        The dataDescribe method provides a statistical description of the data in the form of tables and graphs
        input:
            data: The Hall dataset that is embedded in the package and used to pre train the model
        output:
            A tabular and graphical representation of the statistical analysis of the Hall dataset
            
        """
        data = pd.read_csv(self.cwd+'/TSForecasting/Data/Hall_data.csv') #use your path
        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data['time_gap'] = data['Display Time']- data['Display Time'].shift(1)
        meta = pd.read_csv(self.cwd+'/TSForecasting/Data/Hall_meta.csv')
        data_description = pd.DataFrame()
        
        for subjectId, df in data.groupby('subjectId'):
            subj_id = str(subjectId)
            temp = meta[meta["ID"]==subjectId]
            status = str(temp["status"].values[0])
            l_of_r = df['GlucoseValue'].count()
            maxGV = str(df['GlucoseValue'].max())
            minGV = str(df['GlucoseValue'].min())
            meanGV = round(df['GlucoseValue'].mean(),3)
            miss_val = len(df[df["time_gap"]>str("00:05:10")])
            P_miss_val = round(100*(len(df[df["time_gap"]>str("00:05:00")])/df['GlucoseValue'].count()),2)
            days = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
            start_time = str(df['Display Time'].iloc[0])
            end_time = str(df['Display Time'].iloc[-1])
            temp_df = pd.DataFrame({'Subject ID':[subj_id], 'Status':[status], 'Length of readings':[l_of_r], 'Max. Glucose Value':[maxGV], 'Mean Glucose Value':[meanGV], 'Missing Values':[miss_val], 'Percent of missing values':[P_miss_val], 'Days':[days],'Start':[start_time],'End':[end_time]})
            data_description = pd.concat([temp_df,data_description],ignore_index=True)
            
        display(data_description)

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('Length of the time series\' for all individuals' , fontsize=20)
        plt.xlabel('Subject ID', fontsize=16)
        plt.ylabel('Length of readings', fontsize=16)
        plt.xticks(rotation='vertical')
        ax.bar(data_description['Subject ID'].tolist(),data_description['Length of readings'].tolist())
        plt.show()
        
        days = []
        for i in data_description['Days']:
            days.append(i.days)

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('Days in time series\' for all individuals' , fontsize=20)
        plt.xlabel('Subject ID', fontsize=16)
        plt.ylabel('Days', fontsize=16)
        plt.xticks(rotation='vertical')
        ax.bar(data_description['Subject ID'].tolist(),days)
        plt.show()

        #data_description['Percent of missing values'] = pd.to_numeric(data_description['Percent of missing values'])
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('Percent of missing values for all individuals' , fontsize=20)
        plt.xlabel('Subject ID', fontsize=16)
        plt.ylabel('Percent of missing values', fontsize=16)
        plt.xticks(rotation='vertical')
        ax.bar(data_description['Subject ID'].tolist(),data_description['Percent of missing values'].tolist())
        plt.show()

        #data_description['Missing Values'] = pd.to_numeric(data_description['Missing Values'])
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('Missing Values for all individuals' , fontsize=20)
        plt.xlabel('Subject ID', fontsize=16)
        plt.ylabel('Missing Values', fontsize=16)
        plt.xticks(rotation='vertical')
        ax.bar(data_description['Subject ID'].tolist(),data_description['Missing Values'].tolist())
        plt.show()


        #data_description['Mean Glucose Value'] = pd.to_numeric(data_description['Mean Glucose Value'])
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('Mean Glucose Value for all individuals' , fontsize=20)
        plt.xlabel('Subject ID', fontsize=16)
        plt.ylabel('Mean Glucose Value', fontsize=16)
        plt.xticks(rotation='vertical')
        ax.bar(data_description['Subject ID'].tolist(),data_description['Mean Glucose Value'].tolist())
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(42, 42)

        plt.subplot(3, 2, 1)
        plt.title('Length of the time series\' for all individuals' , fontsize=16)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Missing Values', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.bar(data_description['Subject ID'].tolist(),data_description['Missing Values'].tolist())

        plt.subplot(3, 2, 2)
        plt.title('Days in time series\' for all individuals' , fontsize=16)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Days', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.bar(data_description['Subject ID'].tolist(),days)

        plt.subplot(3, 2, 3)
        plt.title('Percent of missing values for all individuals' , fontsize=16)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Percent of missing values', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.bar(data_description['Subject ID'].tolist(),data_description['Percent of missing values'].tolist())

        plt.subplot(3, 2, 4)
        plt.title('Missing Values for all individuals' , fontsize=16)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Missing Values', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.bar(data_description['Subject ID'].tolist(),data_description['Missing Values'].tolist())

        plt.subplot(3, 2, 5)
        plt.title('Mean Glucose Value for all individuals' , fontsize=16)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Mean Glucose Value', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.bar(data_description['Subject ID'].tolist(),data_description['Mean Glucose Value'].tolist())


        plt.show()

            
    def impute(self,test_data=def_testing):
        """
        The impute method performs the imputations using the trained LSTM model
        input:
            test: testing data
            lstm_model: trainied lstm model
        output:
            A file with imputed values
        """
        test_data = self.datePreprocess(test_data)
        b,e,s,f = self.detectGap(test_data)
        test = test_data.iloc[0:f]

        scaler = MinMaxScaler(feature_range=(0, 1))
        test_sc = scaler.fit_transform(test)
        X_test = test_sc[:-1]
        y_test = test_sc[1:]
        test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)
        for s in range(1,2):
            test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)
    
    
        X_test = test_sc_df.dropna().drop('Y', axis=1)
        y_test = test_sc_df.dropna().drop('X_1', axis=1)
    
        X_test = X_test.as_matrix()
        y_test = y_test.as_matrix()
        
        X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        #print(X_test_lmse)
        y_pred_test_lstm = self.lstm_model.predict(X_test_lmse)
        
        #print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
        
        lstm_test_mse = self.lstm_model.evaluate(X_test_lmse, y_test, batch_size=1)
       
        #print('LSTM: %f'%lstm_test_mse)
        
        #inversing the scaling
        lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
        test_val = scaler.inverse_transform(y_test)

        print("Imputations performed!")
        
        x=0
        for i in range(b-1,e):
            test_data['GlucoseValue'][i] = lstm_pred[x]
            x+=1
        test_data.to_csv(self.cwd+"/TSForecasting/Data/ImputedValues.csv") 
        print("File saved!")
        self.plot(test_data)


    def plot(self, data):
        """
        The plot method plots the graph for the imputed values
        input:
            data: imputed values
        output:
            A plot of the imputed time series
        """
        #plotting true values and lstm predicted values
        #these are original values
        
        plt.figure(figsize=(20, 8))

        plt.plot(data['GlucoseValue'].tolist(), label='True', color='#2280f2', linewidth=2.5)
        
        plt.title("LSTM's Prediction")
        
        plt.xlabel('Observation')
        plt.ylabel('Glucose Values')
        plt.show();


    def detectGap(self, testing_data):
        """
        The detectGap mehtod detects the GAP in a time series
        input:
            testing_data: dataset that needs to be imputed
        output:
            b,s: starting index of the gaps
            e,f: end index of the gaps

        """
        l = []
        k = 0
        for i in testing_data['GlucoseValue']:
            k+=1
            if i==0:
                l.append(k)
        b = min(l)
        e = max(l)
        #print(b,e)
        gap=e-b
        #print(gap)
        #print(b-gap)
        #print(l)
        s = (b-gap-2) if (b-gap) > 0 else 0
        f = b-1
        #print(s,f)
        #print(f-s)
        print("Gap detected!")
        return b,e,s,f


    def index_agreement(self, s,o):
        """
        index of agreement
        input:
            s: prediceted
            o: original
        output:
            ia: index of agreement
        """
        
        ia = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
        
        return ia

    
    def rmse(self, s,o):
        """
        Root Mean Squared Error
        input:
            s: prediceted
            o: original
        output:
            rmses: root mean squared error
        """
        return np.sqrt(np.mean((s-o)**2))

    
    def mae(self, s,o):
        """
        Mean Absolute Error
        input:
            s: prediceted
            o: original
        output:
            maes: mean absolute error
        """
        return np.mean(abs(s-o))

    
    def mad(self, s):
        """
        Mean Absolute Difference
        input:
            s: prediceted
        output:
            mad: mean absolute error
        """
      
        return stats.median_absolute_deviation(s)

    
    def mape(self, y_pred,y_true):
        """
        Mean Absolute Percentage error
        input:
            y_pred: prediceted
            y_true: original
        output:
            mape: Mean Absolute Percentage error
        """
    
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    def fracBias(self, s,o):
        """
        Fractional Bias
        input:
            s: prediceted
            o: original
        output:
            fracBias: Fractional Bias
        """
        
        return np.mean(np.abs((o - s) / ((o + s)/2)))

    
    def getMetrics(self,lstm_pred, test_val):
        """
        The getMetrics method simply prints out all the comparison metrics
        """
        #IOA
        ioa_val = self.index_agreement(lstm_pred,test_val)
        print("Index of Agreement is: " + str(round(ioa_val,3)))
    
        #MAE
        mae_val = self.mae(lstm_pred,test_val)
        print("Mean Absolute Error is: " + str(mae_val))
    
        #RMSE
        rmse_val = self.rmse(lstm_pred,test_val)
        print("Root Mean Squared Error is: " + str(round(rmse_val,3)))
        
        #MAD
        mad_val = self.mad(lstm_pred)
        print("Mean Absolute Difference is: " + str(mad_val))
        
        #FB
        fb_val = self.fracBias(lstm_pred,test_val)
        print("Fractional Bias is: " + str(round(fb_val,3)))
        
        #MAPE
        mape_val = self.mape(lstm_pred,test_val)
        print("Mean Absolute Percentage Error is: " + str(round(mape_val)))
        
    
    def createGap(self, data):
        """
        Creating Gap indexes
        input:
            data: dataframe with index => DisplayTime value => GlucoseValues
        output:
            start: seed
            end: seed+gap (gap=300)
        """
        
        seed = random.randint(500,len(data)-500)
        
        return seed,seed+500
    
    
    def faultyData(self, df,start,end):
        """
        Creating a Gap
        input:
            start: seed
            end: seed+gap (gap=300)
        output:
            df: dataframe with index => DisplayTime value => GlucoseValues and a gap from start to end (inputs)
        """
        
        #df = readData()
        for i in range(start,end):
            df[df.columns[0]][i]=0
        
        return df
    

# =============================================================================
# def summaryPlot(p1,t1,p2,t2,p3,t3):
#     
#     plt.figure(figsize=(20, 20))
# 
#     plt.subplot(3, 1, 1)
#     plt.plot(p1, label='LSTM', color='red', linewidth=2)
#     plt.plot(t1, label='True', color='#2280f2', linewidth=2.5)
#     plt.xlabel('Observation')
#     plt.ylabel('Glucose Values')
#     plt.title("Diabetic - 1")
#     plt.legend()
#     
#     
#     plt.subplot(3, 1, 2)
#     plt.plot(p2, label='LSTM', color='red', linewidth=2)
#     plt.plot(t2, label='True', color='#2280f2', linewidth=2.5)
#     plt.xlabel('Observation')
#     plt.ylabel('Glucose Values')
#     plt.title("Prediabetic")
#     plt.legend()
#     
#     plt.subplot(3, 1, 3)
#     plt.plot(p3, label='LSTM', color='red', linewidth=2)
#     plt.plot(t3, label='True', color='#2280f2', linewidth=2.5)
#     plt.xlabel('Observation')
#     plt.ylabel('Glucose Values')
#     plt.title("Nondiabetic")
#     plt.legend()
#     
#     
#     plt.show();
# =============================================================================
    