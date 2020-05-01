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

    def_testing = pd.read_csv(cwd+'/TSForecasting/Data/test/generated_test.txt', sep=",")

    hall_refined = pd.read_csv(cwd+'/TSForecasting/Data/Hall/Hall_data.csv')

    hall_raw = pd.read_csv(cwd+'/TSForecasting/Data/Hall/data_hall_raw.csv')

    hall_meta = pd.read_csv(cwd+'/TSForecasting/Data/Hall/Hall_meta.csv')

    cgm_appended = pd.read_csv(cwd+'/TSForecasting/Data/CGM/CGM_Analyzer_Appended.csv')

    cgm_original = pd.read_csv(cwd+'/TSForecasting/Data/CGM/CGManalyzer.csv')

    cgm_meta = pd.read_csv(cwd+'/TSForecasting/Data/CGM/CGM-meta.csv')

    def __init__(self):
        """
        The __init__ method initializes and trains the LSTM model on the embedded Hall Dataset
        Input:
            None
        Output:
            A trained model that can be used for imputations
        """
        data = pd.read_csv(self.cwd+'/TSForecasting/Data/CGM/CGManalyzer.csv') #use your path
        #data.head()
        #data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
     
        #dropping columns we don't need
        
        
        data.drop(['subjectId'], axis=1, inplace=True)
        

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
        history_lstm_model = self.lstm_model.fit(X_train_lmse, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
   

    
    def datePreprocess(self,data):
        """
        The datePreprocess method is used to preprocess the testing data
        It indentifies the date and converts it to the standard datetime format
        It also converts the Timestamp to the index
        Input:
            data: dataset we wish to convert the timestamp of {type: pandas DataFrame}
        Output:
            data: dataset with the converted timestamp {type: pandas DataFrame}
        """
        # data = data.reset_index()
        length = data.shape[0]
        for i in range(0,length):
            #print(i)
            s = str(data.iloc[i]['Display Time'])
            k = re.sub("[^0-9]", "", s)
            datetimeObj = parse(k) 
            data = data.replace(to_replace = s, value = datetimeObj)
        
        data = data.set_index(['Display Time'], drop=True)
        
        return(data)


    def train(self, data = cgm_original):
        """
        The train method is used to train the model on user supplied data
        Input:
            data: dataset we want to train the model on {type: pandas DataFrame}
        Output:
            A model trained on the supplied data that can be used for imputations
        """        
        data.drop(['subjectId'], axis=1, inplace=True)
        

        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data = data.set_index(['Display Time'], drop=True)

        # data = self.datePreprocess(data)

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
        history_lstm_model = self.lstm_model.fit(X_train_lmse, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])


    def rawData(self):
        """
        The rawData method provides a statistical description of the raw Hall data in the form of tables and graphs
        This raw data contains large gaps in the time series' of individuals
        It is therefore preprocessed to trim the time series' with smaller gaps and split the time series' with larger gaps
        input:
            None
        output:
            A tabular and graphical representation of the statistical analysis of the raw Hall dataset
            
        """
        # data = self.hall_raw #use your path
        data = self.hall_raw #use your path
        self.clusteringDataDescribe(data)


    def processedData(self):
        """
        The processedData method provides a statistical description of the cleaned Hall data in the form of tables and graphs
        This processed data has large gaps removed in the time series' of individuals by trim the time series' with smaller gaps and split the time series' with larger gaps
        input:
            None
        output:
            A tabular and graphical representation of the statistical analysis of the processed Hall dataset
            
        """
        data = self.hall_refined #use your path
        self.clusteringDataDescribe(data)

    
    def plotSpecific(self,uid,data=cgm_original):
        """
        The plotSpecific method plots the graph of the Glucose Values of a single Subject ID
        Input:
            uid: Subject ID of the user to plot {type: String}
            data: dataset {type: DataFrame}
        Output:
            A plot of the Subject ID's Glucose Values
        """
        # data = self.cgm_original
        new = data[data['subjectId']==str(uid)]
        new = new.astype({'GlucoseValue':int})
        plt.figure(figsize=(20, 8))

        plt.plot(new['Display Time'],new['GlucoseValue'], label='True', color='#2280f2', linewidth=2.5)

        plt.title("Glucose Values of "+str(uid))

        plt.xlabel('Observation')
        plt.ylabel('Glucose Values')
        plt.show();

            
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
        test_data.to_csv(self.cwd+"/TSForecasting/Data/Output/ImputedValues.csv") 
        print("File saved!")
        self.plot(test_data)


    def dataDescribe(self, data = cgm_original, meta = cgm_meta):
        """
        The dataDescribe method provides a statistical description of the CGM Analyzer data in the form of tables and graphs
        This processed data has large gaps removed in the time series' of individuals by trim the time series' with smaller gaps and split the time series' with larger gaps
        input:
            data: CGM Analyzer data {DataFrame}
            meta: CGM Analyzer data metadata{DataFrame}
        output:
            A tabular and graphical representation of the statistical analysis of the processed Hall dataset
            
        """
        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        print("Here is a glimpse of the data:\n")
        print(data.head())
        print("\n\n\n")
        
        data_description = pd.DataFrame()
        
        for subjectId, df in data.groupby('subjectId'):
            df['time_gap'] = df['Display Time']- df['Display Time'].shift(1)
            subj_id = str(subjectId)
            temp = meta[meta["ID"]==subj_id]
            status = str(temp["status"].values[0])
            l_of_r = df['GlucoseValue'].count()
            maxGV = round(df['GlucoseValue'].max(),2)
            minGV = round(df['GlucoseValue'].min(),2)
            # gap_size = df[df['time_gap']>str("00:03:10")]
            # gap_size = max(gap_size.time_gap)
            days = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
            start_time = str(df['Display Time'].iloc[0])
            end_time = str(df['Display Time'].iloc[-1])
            temp_df = pd.DataFrame({'Subject ID':[subj_id], 'Status':[status], 'Length of readings':[l_of_r], 'Max. Glucose Value':[maxGV], 'Min. Glucose Value':[minGV],  'Days':[days],'Start':[start_time],'End':[end_time]})
            data_description = pd.concat([temp_df,data_description],ignore_index=True)

        temp = None



        display(data_description.describe())

        print("Here is the statistical analysis of the data:\n")
        display(data_description)
        print("\n\n")

        days = []
        for i in data_description['Days']:
            days.append(i.days)


        fig = plt.figure()
        fig.set_size_inches(36, 36)
        fig.suptitle("Graphical Analysis of data")

        plt.subplot(2, 1, 1)
        plt.title('Length of the time series\' for all individuals' , fontsize=16)
        plt.xlabel('Length', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(data_description['Length of readings'].tolist())

        plt.subplot(2, 1, 2)
        plt.title('Days in time series\' for all individuals' , fontsize=16)
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(days)


        plt.show()


#==================================================================================================================
    def clusteringDataDescribe(self,data):
        
        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        print("Here is a glimpse of the data:\n")
        print(data.head())
        print("\n\n\n")
        meta = self.hall_meta
        data_description = pd.DataFrame()
        
        for subjectId, df in data.groupby('subjectId'):
            df['time_gap'] = df['Display Time']- df['Display Time'].shift(1)
            subj_id = str(subjectId)
            temp = meta[meta["ID"]==subj_id]
            status = str(temp["status"].values[0])
            l_of_r = df['GlucoseValue'].count()
            maxGV = str(df['GlucoseValue'].max())
            minGV = str(df['GlucoseValue'].min())
            gap_size = df[df['time_gap']>str("00:05:10")]
            gap_size = max(gap_size.time_gap)
            miss_val = len(df[df["time_gap"]>str("00:05:10")])
            P_miss_val = round(100*(len(df[df["time_gap"]>str("00:05:10")])/df['GlucoseValue'].count()),2)
            days = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
            start_time = str(df['Display Time'].iloc[0])
            end_time = str(df['Display Time'].iloc[-1])
            temp_df = pd.DataFrame({'Subject ID':[subj_id], 'Status':[status], 'Length of readings':[l_of_r], 'Max. Glucose Value':[maxGV], 'Min. Glucose Value':[minGV], 'Gapsize':[gap_size], 'Missing Values':[miss_val], 'Percent of missing values':[P_miss_val], 'Days':[days],'Start':[start_time],'End':[end_time]})
            data_description = pd.concat([temp_df,data_description],ignore_index=True)

        temp = None



        display(data_description.describe())

        print("Here is the statistical analysis of the data:\n")
        display(data_description)
        print("\n\n")

        days = []
        for i in data_description['Days']:
            days.append(i.days)

        gaps = []
        for i in data_description['Gapsize']:
            gaps.append(i.days)


        fig = plt.figure()
        fig.set_size_inches(36, 36)
        fig.suptitle("Graphical Analysis of data")

        plt.subplot(3, 2, 1)
        plt.title('Length of the time series\' for all individuals' , fontsize=16)
        plt.xlabel('Length', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(data_description['Length of readings'].tolist())

        plt.subplot(3, 2, 2)
        plt.title('Days in time series\' for all individuals' , fontsize=16)
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(days)

        plt.subplot(3, 2, 3)
        plt.title('Percent of missing values for all individuals' , fontsize=16)
        plt.xlabel('Percent of missing values', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(data_description['Percent of missing values'].tolist())

        plt.subplot(3, 2, 4)
        plt.title('Missing Values for all individuals' , fontsize=16)
        plt.xlabel('Missing Values', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(data_description['Missing Values'].tolist())

        plt.subplot(3, 2, 5)
        plt.title('Gaps in time series\' for all individuals' , fontsize=16)
        plt.xlabel('Gaps (in days)', fontsize=12)
        plt.ylabel('No. of Individuals', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.hist(gaps)
        plt.show()


        plt.show()

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
        

    
