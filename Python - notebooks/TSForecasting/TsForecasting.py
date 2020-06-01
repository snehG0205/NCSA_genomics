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

from TSForecasting.mage_calc import MageDataSet


from scipy import stats

import random
import re
from dateutil.parser import parse

import warnings  
warnings.filterwarnings('ignore')

import os


class TimeSeriesForecast:


#==================================================================================================================
#   Local variables
#==================================================================================================================

    lstm_model = None
    
    cwd = os.getcwd()

    # hall_refined = pd.read_csv(cwd+'/TSForecasting/Data/Hall/Hall_data.csv')

    # hall_raw = pd.read_csv(cwd+'/TSForecasting/Data/Hall/data_hall_raw.csv')

    # hall_meta = pd.read_csv(cwd+'/TSForecasting/Data/Hall/Hall_meta.csv')

    # cgm_appended = pd.read_csv(cwd+'/TSForecasting/Data/CGM/CGM_Analyzer_Appended.csv')

    # cgm_original = pd.read_csv(cwd+'/TSForecasting/Data/CGM/CGManalyzer.csv')

    # cgm_meta = pd.read_csv(cwd+'/TSForecasting/Data/CGM/CGM-meta.csv')

    # gluvarpro = pd.read_csv(cwd+'/TSForecasting/Data/Gluvarpro/Gluvarpro.csv')

    # gluvarpro_meta = pd.read_csv(cwd+'/TSForecasting/Data/Gluvarpro/GVP_metadata.csv')

    # ohio_full = pd.read_csv(cwd+'/TSForecasting/Data/Ohio-Data/OhioFullConsolidated.csv')

    # ohio_meta = pd.read_csv(cwd+'/TSForecasting/Data/Ohio-Data/Ohio_metadata.csv')

    consolidated_paper = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPaper.csv')

    consolidated_pkg = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPackage.csv')

    consolidated_meta = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedMetadata.csv')

    def_training = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPaper.csv')


#==================================================================================================================
#   Core Methods
#==================================================================================================================

    def __init__(self):
        """
            Package name: TSForecasting
            
            Class name: TimeSeriesForecast
            
            filename: TsForecasting
            
            Import code: from TSForecasting.TsForecasting import TimeSeriesForecast

            Creating an object: object = TimeSeriesForecast()

            Methods:

			+-------------------------------------------------------------------------------------------------------------------+
			| 	Method Name 	|				Description				|			Input			|			Output			|
			+-------------------------------------------------------------------------------------------------------------------+	
			|		init		|	The __init__ method initializes	the |			None			|			None			|
			|					|	data frames and testing functions 	|							|							|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|	datePreprocess	|	The datePreprocess method is used	|	data: dataset we wish	|	data: dataset with the 	|
			|					|	to preprocess the testing data.		|	to convert the time - 	|	converted timestamp		|
			|					|	It identifies the date and converts |	stamp of {type: 		|	{type: dataframe}		|
			|					|	it to the standard datetime format.	|	dataframe}				|							|
			|					|	It also converts the Timestamp 		|							|							|
			|					|	to the index 						|							|							|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|		train		|	The train method is used to train	|	data: dataset we want	|	A trained model that	|
			|					|	the model on user supplied data 	|	to train the model on	|	can be used for 		|
			|					|	 									|	{type: dataframe}		|	imputations				|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|		impute		|	The impute method performs the		|	test: dataset {type:	|	A file with imputed 	|
			|					|	imputations using the trained LSTM 	|	dataframe}			 	|	values				 	|
			|					|	model 								|	lstm_model: trainied 	|							|
			|					|										|	lstm model				|							|
			|					|	 									|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|	plotSpecific	|	The plotSpecific method plots the	|	uid: Subject ID of the 	|	A plot of the patient's	|
			|					|	graph of the Glucose Values of a 	|	user to plot {type:		|	Glucose values 			|
			|					|	single patient 						|	String}					|							|
			|					|										|	data: dataset {tye: 	|							|
			|					|										|	DataFrame}				|							|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|	dataDescribe	|	The dataDescribe method provides &	|	data: CGM Analyzer data	|	A tabular and graphical |
			|					|	statistical description of the CGM 	|	{tye: DataFrame}	 	|	representation of the	|
			|					|	Analyzer data in the form of tables	|	meta: CGM Analyzer data |	statistical analysis of	|
			|					|	and graphs. This processed data has |	metadata {type: 		|	the CGM Anayzer dataset	|
			|					|	large gaps removed in the time 		|	DataFrame}				|							|
			|					|	series' of individuals by trim the 	|							|							|
			|					|	time series' with smaller gaps and	|							|							|
			|					|	split the time series' with larger 	|							|							|
			|					|	gaps  								|							|							|
			|					|	 									|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|	smoothing		|	Performing rolling-mean smoothening |	data: time-series with 	|	data: smoothened    	|
			|					|	for a time-series to improve  		|	irregular intervals 	|	 time-series			|
			|					|	MAGE calculation					|	{type: dataframe}		|	{type: dataframe}		|
			|					|	 									|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|	subSample		|	Sampling a time-series at 15 		|	data: dataset we want	|	data: time-series with 	|
			|					|	minute intervals 					|	to train the model on	|	15 minute intervals 	|
			|					|	 									|	{type: dataframe}		|	{type: dataframe}		|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|  convertUnits		|	converting glucose values into 		|	data: individual time 	|	data: individual time 	|
			|					|	desired unit of measurement		 	|	series with default 	|	series desired unit		|
			|					|	 									|	unit of glucose values 	|	of glucose value		|
			|					|										|	{type: dataframe}		|	{type: dataframe}		|
			|					|										|							|							|
			|					|										|	unit: desired unit of 	|							|
			|					|										|	conversion				|							|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			|	full_days		|	trimming an individual's glucose 	|	data: irregular 	 	|	data: time series with 	|
			|					|	values to only consist of 		 	|	time-series				|	data only for full days	|
			|					|	full days 							|	{type: dataframe}		|	imputations				|
			|					|										|							|	{type: dataframe}		|
			|					|										|							|							|
			+-------------------------------------------------------------------------------------------------------------------+
			  

            Variables:

			+---------------------------------------------------------------+
			| 	Variable Name 		|			Data it contains			|
			+---------------------------------------------------------------+
			|	consolidated_paper	|		The consolidated data from  	|
			|						|		CGMAnalyzer,CGMAnalysis,		|
			|						|		GluVarPro, and Ohio dataset 	|
			|						|										|
			+---------------------------------------------------------------+
			|	consolidated_pkg	|		The consolidated data from  	|
			|						|		CGMAnalyzer,CGMAnalysis,		|
			|						|		GluVarPro, Ohio, and Hall 	 	|
			|						|		dataset							|
			|						|										|
			+---------------------------------------------------------------+
			|	consolidated_meta	|		The metadata for consolidated 	|
			|						|		data							|
			|						|										|
			+---------------------------------------------------------------+
            


            Package dependencies:
                - pandas
                - numpy
                - matplotlib
                - dateutil
                - re     
        """
        print("Object Created!")
        
    
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


    def train(self, data = def_training):
        """
            The train method is used to train the model on user supplied data
            Input:
                data: dataset we want to train the model on {type: pandas DataFrame}
            Output:
                A model trained on the supplied data that can be used for imputations
        """ 
        print("Training Model...\n\n")       
        data.drop(['subjectId'], axis=1, inplace=True)
        

        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data = data.set_index(['Display Time'], drop=True)
        # data = self.datePreprocess(data)

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
       
    
        # print('Train shape: ', X_train_lmse.shape)
        
    
        self.lstm_model = Sequential();
        self.lstm_model.add(LSTM(7, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False));
        self.lstm_model.add(Dense(1));
        self.lstm_model.compile(loss='mean_squared_error', optimizer='adam');
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1);
        history_lstm_model = self.lstm_model.fit(X_train_lmse, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop]);
        print("Model traines successfully!")


    def plotSpecific(self,uid,data= consolidated_paper):
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

        plt.plot(new['Display Time'],new['GlucoseValue'], color='#2280f2', linewidth=2.5)

        plt.title("Glucose Values of "+str(uid))

        plt.xlabel('Observation')
        plt.ylabel('Glucose Values')
        plt.show();

            
    def impute(self,test_data,flag=0):
        """
            The impute method performs the imputations using the trained LSTM model
            Input:
                test_data: testing data
                lstm_model: trainied lstm model
            Output:
                A file with imputed values
        """
        test_data = self.datePreprocess(test_data)
        b,e,s,f,gaps = self.detectGap(test_data)
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
        y_pred_test_lstm = self.lstm_model.predict(X_test_lmse);
        
        #print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
        
        
        #inversing the scaling
        lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
        test_val = scaler.inverse_transform(y_test)

        lstm_pred = lstm_pred.tolist()
        
        lstm_pred = lstm_pred*200
        
        x=0
        for i in range(b-1,e):
            test_data['GlucoseValue'][i] = lstm_pred[x][0]
            x+=1
        

        if flag==1:
            return test_data
        else:
            print("Imputations performed!")
            test_data.to_csv(self.cwd+"/TSForecasting/Data/Output/ImputedValues.csv") 
            print("File saved!\nLocation:"+str(self.cwd+"/TSForecasting/Data/Output/ImputedValues.csv"))
            self.plot(test_data)


    def dataDescribe(self, data = consolidated_paper, meta = consolidated_meta):
        """
            The dataDescribe method provides a statistical description of the CGM Analyzer data in the form of tables and graphs
            This processed data has large gaps removed in the time series' of individuals by trim the time series' with smaller gaps and split the time series' with larger gaps
            Input:
                data: CGM Analyzer data {DataFrame}
                meta: CGM Analyzer data metadata{DataFrame}
            Output:
                A tabular and graphical representation of the statistical analysis of the processed Hall dataset
            
        """
        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        print("Here is a glimpse of the data:\n")
        print(data.head())
       
    
        total_readings = data['Display Time'].count()
        print("\nTotal Readings in the data:"+str(total_readings))
        print("\n\n\n")

        data_description = pd.DataFrame()
        
        for subjectId, df in data.groupby('subjectId'):
            
            subj_id = str(subjectId)

            # df = self.full_days(df)
            
            df['time_gap'] = df['Display Time']- df['Display Time'].shift(1)
            days = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
            
            
            temp = meta[meta["ID"]==subj_id]
            status = str(temp["status"].values[0])
            
            l_of_r = df['GlucoseValue'].count()
            
            maxGV = round(df['GlucoseValue'].max(),2)
            minGV = round(df['GlucoseValue'].min(),2)
            
            # smoothened = self.smoothing(df['GlucoseValue'])
            glucs = df['GlucoseValue'].to_list()
            indices = [1*i for i in range(len(glucs))]
            m = MageDataSet(indices, glucs)
            k = m.getMAGE()
            # gap_size = df[df['time_gap']>str("00:03:10")]
            # gap_size = max(gap_size.time_gap)

            start_time = str(df['Display Time'].iloc[0])
            end_time = str(df['Display Time'].iloc[-1])
            
            temp_df = pd.DataFrame({'Subject ID':[subj_id], 'Length of readings':[l_of_r], 'Max. Glucose Value':[maxGV], 'Min. Glucose Value':[minGV], 'MAGE Score':[k], 'Days':[days], 'Start':[start_time],'End':[end_time]})
            data_description = pd.concat([temp_df,data_description],ignore_index=True)

        temp = None

        data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)


        display(data_description.describe())

        print("Here is the statistical analysis of the data:\n")
        display(data_description)
        print("\n\n")

        data_description.to_csv(self.cwd+"/TSForecasting/Data/Data Description.csv")


        # days = []
        # for i in data_description['Days']:
        #     days.append(i.days)


        # fig = plt.figure()
        # fig.set_size_inches(36, 36)
        # fig.suptitle("Graphical Analysis of data")

        # plt.subplot(2, 1, 1)
        # plt.title('Length of the time series\' for all individuals' , fontsize=32)
        # plt.xlabel('Length', fontsize=24)
        # plt.ylabel('No. of Individuals', fontsize=24)
        # plt.xticks(rotation='vertical')
        # plt.rc('xtick',labelsize=18)
        # plt.rc('ytick',labelsize=18)
        # plt.hist(data_description['Length of readings'].tolist())

        # plt.subplot(2, 1, 2)
        # plt.title('Days in time series\' for all individuals' , fontsize=32)
        # plt.xlabel('Days', fontsize=24)
        # plt.ylabel('No. of Individuals', fontsize=24)
        # plt.xticks(rotation='vertical')
        # plt.rc('xtick',labelsize=18)
        # plt.rc('ytick',labelsize=18)
        # plt.hist(days)


        # plt.show()


#==================================================================================================================
#   Helper Methods
#==================================================================================================================

    def smoothing(self, data):
        """
            Performing rolling-mean smoothening for a time-series to improve MAGE calculation 
            Input:
                data: time-series with tiny fluctuations and local minima and maxima
            Output:
                data: smoothened time-series
        """ 
        
        series = data
        # Tail-rolling average transform
        rolling = series.rolling(window=6)
        rolling_mean = rolling.mean()
        return rolling_mean


    def subSample(self, data):
        """
            Sampling a time-series at 15 minute intervals
            Input:
                data: time-series with irregular intervals
            Output:
                data: time-series with 15 minute intervals
        """ 
        data['Display Time'] = pd.to_datetime(data['Display Time'])
        data['time_gap'] = data['Display Time'].shift(1)-data['Display Time'][0]
        data['time_gap'][0] = '00:00:00'
        mods = [0,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899]
        subset = pd.DataFrame()
        for i in range(1,len(data.index)):
            seconds = data['time_gap'][i].total_seconds()
            if (seconds%900) in mods:
                subj_id = data['subjectId'][i]
                gv = data['GlucoseValue'][i]
                dt = data['Display Time'][i]
                temp_df = pd.DataFrame({'Display Time':[dt], 'GlucoseValue':[gv], 'subjectId':[subj_id]})
                subset = pd.concat([temp_df,subset],ignore_index=True)
        subset = subset.iloc[::-1]
        subset = subset.reset_index(drop=True)
        data.drop(['time_gap'], axis=1, inplace=True)
        return subset


    def convertUnits(self, data, unit):
        """
            converting glucose values into desired unit of measurement 
            Input:
                data: individual time series with default unit of glucose value
                unit: desired unit of conversion
            Output:
                data: individual time series desired unit of glucose value
        """ 
        if unit == 'mmol':
            if(data['GlucoseValue'][0]>18):
                in_mmols = pd.DataFrame({'Display Time':data['Display Time'], 'GlucoseValue':data['GlucoseValue']/18, 'subjectId':data['subjectId']})
                return in_mmols
            else:
                print("Data already in mmols")
                return data
        elif unit == 'mg':
            if(data['GlucoseValue'][0]<18):
                in_mgs = pd.DataFrame({'Display Time':data['Display Time'], 'GlucoseValue':data['GlucoseValue']*18, 'subjectId':data['subjectId']})
                return in_mgs
            else:
                print("Data already in mgs")
                return data
        else:
            print("Invalid unit. Please enter 'mmol' or 'mg'. ")
            
            
    def full_days(self, data):
        """
           trimming an individual's glucose values to only consist of full days     
            Input:
                data: irregular time series
            Output:
                data: time series with data only for full days
        """ 
        
        dates = list()
        for i in range(0,len(data.index)):
            dates.append(data['Display Time'][i].date())
        data['Dates'] = dates
        end = data['Dates'].iloc[-1]
        start = data['Dates'].iloc[0]

        indexVals = data[ data['Dates'] == start ].index
        # indexVals
        data.drop(indexVals , inplace=True)

        indexVals = data[ data['Dates'] == end ].index
        # indexVals
        data.drop(indexVals , inplace=True)

        data = data.reset_index(drop=True)
        
        data.drop(['Dates'], axis=1, inplace=True)
        
        return data

        
#==================================================================================================================
#   These methods are not called by the user and only used for internal processing
#==================================================================================================================
    
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
        # print("Gap detected!")
        return b,e,s,f,gap


#==================================================================================================================
#   Calculating error metrics
#==================================================================================================================

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

    
    def mad(self, s,o):
        """
        Mean Absolute Error
        input:
            s: prediceted
            o: original
        output:
            maes: mean absolute difference
        """
        return np.mean(abs(s-o))

    
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
 

#==================================================================================================================
#   GVI's
#==================================================================================================================

    def gvIndices(self, data = consolidated_paper):
        data_description = pd.DataFrame()
        for subjectId, df in data.groupby('subjectId'):
        #     print(subjectId)
            df['Display Time'] = pd.to_datetime(df['Display Time'])
            df=df.reset_index(drop=True)
            gfi, gcf = self.gfi(df)
            LBGI, HBGI, BGRI = self.bgri(df, units = 'mg')
            GRADE , HypoG_P, EuG_P, HyperG_P = self.grade(df, units='mg')
            j_index = self.j_index(df, units="mg")
            Mvalue = self.m_value(df, 'mg', 120)
            MAG = self.mag(df)
            GVP = self.gvp(df, units='mg')
            GMI = self.gmi(df, units='mg')
            LAGE, MAX, MIN = self.lage(df)
            HBA1C = self.ehba1c(df)
            m, sd, cv, iqr = self.sumstats(df)
            sdrc = self.rc(df)
            temp_df = pd.DataFrame({'Subject ID':[subjectId], 'GFI':[gfi], 'GCF':[gcf], 'LBGI':[LBGI], 'HBGI':[HBGI], 'BGRI':[BGRI], 'GRADE':[GRADE], 'HypoG_P':[HypoG_P],'EuG_P':[EuG_P], 'HyperG_P':[HyperG_P], 'J Index':[j_index], 'Mvalue':[Mvalue], 'MAG':[MAG], 'GVP':[GVP], 'GMI':[GMI], 'LAGE':[LAGE],'MAX':[MAX], 'MIN':[MIN], 'HBA1C':[HBA1C], 'MEAN':[m], 'STD-DEV':[sd],'CV':[cv], 'IQR':[iqr], 'SDRC':[sdrc]})
            data_description = pd.concat([temp_df,data_description],ignore_index=True)

        data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)

        display(data_description)

        data_description.to_csv(self.cwd+"/TSForecasting/Data/Glucose Indices.csv")


    def gfi(self, x):
        N = len(x)
        S = 0
        for i in range(0,N-1):
            S = S + (x.iloc[i, 1]  - x.iloc[(i+1), 1]) ** 2
            
        gfi = np.sqrt(S/N)
        gcf = gfi/np.mean(x.iloc[:,1])
        # return pd.DataFrame({'GFI':[gfi], 'GCF':[gcf]})
        return gfi, gcf


    def bgri(self, x, units):
        if (units == 'mg'):
            fBG = 1.509*((np.log(   x.iloc[:, 1]) )**1.084  - 5.381)
        elif (units=='mmol'):
            fBG = 1.509*((np.log(18*x.iloc[:, 1]) )**1.084  - 5.381)
        else:
            print('units should be either mmol or mg')
            return 0
            
        rBG = 10 * fBG ** 2 # called BG risk function
        s = np.sign(fBG)
        s_left = np.abs(s.where(s == -1, 0))
        rlBG = rBG * s_left # called BG risk function left branch

        s_right = s.where(s == 1, 0)
        rhBG = rBG * s_right # called BG risk function right branch

        LBGI = np.mean(rlBG)#1/len(rlBG)*np.sum(rlBG) # low BD index
        HBGI = np.mean(rhBG)#1/len(rhBG)*np.sum(rhBG) # high BD index
        BGRI = (LBGI + HBGI) # BG risk index
              
        # return pd.DataFrame({'LBGI':[LBGI], 'HBGI':[HBGI], 'BGRI':[BGRI]})
        return LBGI, HBGI, BGRI


    def grade(self, x, units):
        if (units == 'mg'):
            a = 18
            g = np.append(np.where(x.iloc[:, 1] <= 37)[0], np.where(x.iloc[:, 1] >= 630)[0])
            hypo = np.where(x.iloc[:, 1] < 70)[0]
            eu = np.where((x.iloc[:, 1] >= 70) & (x.iloc[:, 1]<=140))[0]
            hyper = np.where(x.iloc[:, 1] > 140)[0]
        elif (units=='mmol'):
            a = 1
            g = np.append(np.where(x.iloc[:, 1] <= 2.06)[0], np.where(x.iloc[:, 1] >= 33.42)[0])
            hypo = np.where(x.iloc[:, 1]<3.9)[0]
            eu = np.where(x.iloc[:, 1]>=3.9 & x.iloc[:, 1] <=7.8)[0]
            hyper = np.where(x.iloc[:, 1]>7.8)[0]
        else:
            print('units should be either mmol or mg')
            return 0
        
        grd = 425*( np.log10( np.log10(a*x.iloc[:, 1]) ) + 0.16) ** 2

      
        if (len(g)>0):  # GRADE is designed to operate for BG ranges between 2.06 (37 mg/dl) and 33.42 mmol/l (630 mg/dl).
            grd[g] = 50 # Values outside this range are ascribed a GRADE value of 50.

        tmp = (np.mean(grd), len(hypo)/len(x)*100, len(eu)/len(x)*100, len(hyper)/len(x)*100)

        GRADE = np.mean(grd)
        HypoG_P = len(hypo)/len(x)*100
        EuG_P = len(eu)/len(x)*100
        HyperG_P = len(hyper)/len(x)*100
        
        # return pd.DataFrame({'GRADE':[np.mean(grd)], 'HypoG%':[len(hypo)/len(x)*100], 'EuG%':[len(eu)/len(x)*100], 'HyperG%':[len(hyper)/len(x)*100]})
        return GRADE , HypoG_P, EuG_P, HyperG_P

    
    def j_index(self, x, units):
        if (units == 'mg'):
            a = 0.001
        elif (units=='mmol'):
            a = 0.324
        else:
            print('units should be either mmol or mg')
            return 0
        
        j = a*(np.mean(x.iloc[:, 1]) + np.std(x.iloc[:, 1])) ** 2
        
        # return pd.DataFrame({'J-index':[j]})
        return j


    def m_value(self, x, units, ref_value):
        if (units == 'mg'):
            PG = x.iloc[:, 1]
        elif (units=='mmol'):
            PG = 18*x.iloc[:, 1]
        else:
            print('units should be either mmol or mg')
            return 0
        
        if ((ref_value != 120) & (ref_value != 90) & (ref_value != 80) ):
            print('ref_value should be set to one of these: 80, 90, 120')
            return 0
        
        M_BSBS = np.abs((10*np.log(PG/ref_value))**3)

        if (len(PG)<25):
            W = np.max(PG) - np.min(PG)
            Mvalue = np.mean(M_BSBS) + W/20 
        else:
            Mvalue = np.mean(M_BSBS)

        # return pd.DataFrame({'M-value':[Mvalue]})
        return Mvalue


    def mag(self, x):
        S = np.abs(np.sum(x.iloc[:, 1].diff()))
        n = len(x)-1
        total_T = (x.iloc[n,0] - x.iloc[0, 0])/np.timedelta64(1,'h')
        MAG = S/total_T
        # return pd.DataFrame({'MAG':[MAG]})
        return MAG

    
    def gvp(self, x, units):
        if (units != 'mg'):
            print('units can only be mg')
            return 0
        
        dt = x.iloc[:, 0].diff()/np.timedelta64(1,'m') # assuming that sampling can not necessarily be equally spaced
        dy = x.iloc[:, 1].diff()
        
        L = np.sum(np.sqrt(dt**2 + dy**2))
        L_0 = np.sum(dt)
        
        GVP = (L/L_0 -1) *100
        # return pd.DataFrame({'GVP(%)':[GVP]})
        return GVP


    def gmi(self, x, units):
        if (units == 'mg'):
            GMI = 3.31 + 0.02392 * np.mean(x.iloc[:, 1])
            # return pd.DataFrame({'GMI(%)': [GMI]})
            return GMI
        elif (units=='mmol'):
            GMI = 12.71 + 4.70587 * np.mean(x.iloc[:, 1])
            # return pd.DataFrame({'GMI(%)': [GMI]})
            return GMI
        else:
            print('units should be either mmol or mg')
            return 0
    

    def lage(self, x):
        MIN = np.min(x.iloc[:, 1])
        MAX = np.max(x.iloc[:, 1])
        LAGE = MAX - MIN
        # return pd.DataFrame({'LAGE': [LAGE], 'MAX': [MAX], 'MIN':[MIN]})
        return LAGE, MAX, MIN


    def ehba1c(self, x):
        HBA1C = (np.mean(x.iloc[:, 1]) + 46.7)/28.7
        # return pd.DataFrame({'eHbA1c': [HBA1C]})
        return HBA1C


    def sumstats(self, x):
        m = np.mean(x.iloc[:, 1])
        sd = np.std(x.iloc[:, 1])
        cv = sd/m
        q75, q25 = np.percentile(x.iloc[:, 1], [75 ,25])
        iqr = q75 - q25
        
        # return pd.DataFrame({'Mean': [m], 'SD':[sd], 'CV': [cv], 'IQR': [iqr]})
        return m, sd, cv, iqr


    def rc(self, x):
        dt = x.iloc[:, 0].diff()/np.timedelta64(1,'m') 
        dy = x.iloc[:, 1].diff()
        
        sdrc = np.std(dy/dt)
        # return pd.DataFrame({'SD of RC': [sdrc]})
        return sdrc

    
    def pgs(self, x, units):
        # if (units != 'mg'):
        #     return print('units can only be mg')
        
        # GVP = gvp(x, units=units)
        # F_GVP = 1 + 9/(1 + np.exp(-0.049*(GVP-65.47)))
        
        # MG = np.mean(x.iloc[:, 3])
        # F_MG = 1 + 9 * ( 1/(1 + np.exp(0.1139*(MG-72.08))) + 1/(1 + np.exp(-0.09195*(MG-157.57))) )
        
        # N54 = 
        # F_54H = 0.5 + 4.5 * (1 - np.exp(-0.81093*N54))
        return 0



    
    
