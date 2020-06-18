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

import math

from statistics import mean

from datetime import timedelta
from datetime import datetime


class TimeSeriesForecast:


#==================================================================================================================
#   Local variables
#==================================================================================================================

    lstm_model = None
    
    cwd = os.getcwd()

    consolidatedData = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPaper.csv')

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

            Package dependencies:
                - pandas
                - numpy
                - matplotlib
                - dateutil
                - re     

            Read complete documentation here:
            https://wiki.ncsa.illinois.edu/display/CPRHD/Package+Documentation
        """
        

        # display(self.consolidatedData)


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
        data = self.fullDaysOnly(data)     
        data.drop(['subjectId'], axis=1, inplace=True)
        

        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
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
        print("Model trained successfully!")


    def plotSpecific(self, uid, data= consolidatedData):
        """
            The plotSpecific method plots the graph of the Glucose Values of a single Subject ID
            Input:
                uid: Subject ID of the user to plot {type: String}
                data: dataset {type: DataFrame}
            Output:
                A plot of the Subject ID's Glucose Values
        """
        new = data[data['subjectId']==str(uid)]
        new = new.astype({'GlucoseValue':int})
        plt.figure(figsize=(20, 8))

        plt.plot(new['GlucoseValue'], color='#2280f2', linewidth=2.5)

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
        # subj_id = test_data['subjectId'][0]
        # test_data.drop(['subjectId'], axis=1, inplace=True)
        test_data = self.datePreprocess(test_data)
        b,e,s,f,gaps = self.detectGap(test_data)
        test = test_data.iloc[0:f]
        test.drop(['subjectId'], axis=1, inplace=True)
        

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
            # test_data['subjectId'] = subj_id
            test_data.to_csv(self.cwd+"/TSForecasting/Data/Output/ImputedValues.csv") 
            print("File saved!\nLocation:"+str(self.cwd+"/TSForecasting/Data/Output/ImputedValues.csv"))
            self.plot(test_data)


    def dataDescribe(self, data = consolidatedData, meta = consolidated_meta):
        """
            The dataDescribe method provides a statistical description of the CGM Analyzer data in the form of tables and graphs
            This processed data has large gaps removed in the time series' of individuals by trim the time series' with smaller gaps and split the time series' with larger gaps
            Input:
                data: CGM Analyzer data {DataFrame}
                meta: CGM Analyzer data metadata{DataFrame}
            Output:
                A tabular and graphical representation of the statistical analysis of the processed Hall dataset
            
        """
        data = self.fullDaysOnly(data)
        data['Display Time'] = data['Display Time'].apply(lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
        # print("Here is a glimpse of the data:\n")
        # print(data.head())
       
    
        total_readings = len(data.index)
        print("\nTotal Readings in the data:"+str(total_readings))
        print("\n\n\n")

        data_description = pd.DataFrame()
        
        for subjectId, df in data.groupby('subjectId'):
            
            day = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
            day = day.round("d")
            day = day.days
            
            temp = meta[meta["ID"]==str(subjectId)]
            status = str(temp["status"].values[0])
            
            l_of_r = df['GlucoseValue'].count()
            
            maxGV = round(df['GlucoseValue'].max(),2)
            minGV = round(df['GlucoseValue'].min(),2)

            df = df.reset_index(drop=True)
            mean_day, mean_night = self.meanCalculations(df)
            
            # smoothened = self.smoothing(df['GlucoseValue'])
            df['GlucoseValue'] = self.smoothing(df['GlucoseValue'])
            df = df[df['GlucoseValue'].notna()]
            df = df.reset_index(drop=True)

            dates = []
            for i in range(len(df.index)):
                dates.append(df['Display Time'][i].date())
            df['Date'] = dates

            mage_daily = []
            excursions = 0
            for Date, xx in df.groupby('Date'):
                xx = xx.reset_index(drop=True)
                m, e = self.mageCalculation(xx,1)
                mage_daily.append(m)
                excursions = excursions + e

            mage = round(mean(mage_daily),3)
            excusrions = math.floor(excursions)

            
            temp_df = pd.DataFrame({'Subject ID':[subjectId], 'Length of readings':[l_of_r], 'Max. Glucose Value':[maxGV], 'Min. Glucose Value':[minGV], 'MAGE Score':[mage], "Excursions":[excursions],'Days':[day], 'Daytime Mean:':[mean_day], 'Nighttime Mean':[mean_night]})
            data_description = pd.concat([data_description, temp_df],ignore_index=True)

        # data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)

        print("Here is the statistical analysis of the data:\n")
        display(data_description)
        print("\n\n")

        data_description.to_csv(self.cwd+"/TSForecasting/Data/Data Description.csv")


    def individualDescribe(self, uid, data = consolidatedData, meta = consolidated_meta):
        df = data[data['subjectId']==str(uid)]
        
        df = self.fullDaysOnly(df)
        df = df.reset_index(drop=True)
        df['Display Time'] = df['Display Time'].apply(lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))


        day = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
        day = day.round("d")
        day = day.days

        
        if uid in meta["ID"]:
            temp = meta[meta["ID"]==uid]
            status = str(temp["status"].values[0])
        else:
            status = "Undefined"
        
        l_of_r = df['GlucoseValue'].count()
        
        maxGV = round(df['GlucoseValue'].max(),2)
        minGV = round(df['GlucoseValue'].min(),2)


        df = df.reset_index(drop=True)
        mean_day, mean_night = self.meanCalculations(df)
        
        # smoothened = self.smoothing(df['GlucoseValue'])
        df['GlucoseValue'] = self.smoothing(df['GlucoseValue'])
        df = df[df['GlucoseValue'].notna()]
        df = df.reset_index(drop=True)

        dates = []
        for i in range(len(df.index)):
            dates.append(df['Display Time'][i].date())
        df['Date'] = dates

        mage_daily = []
        excursions = 0
        for Date, xx in df.groupby('Date'):
            xx = xx.reset_index(drop=True)
            m, e = self.mageCalculation(xx,1)
            mage_daily.append(m)
            excursions = excursions + e

        mage = round(mean(mage_daily),3)
        excusrions = math.floor(excursions)
        
        data_description = pd.DataFrame({'Subject ID':[uid], 'Length of readings':[l_of_r], 'Max. Glucose Value':[maxGV], 'Min. Glucose Value':[minGV], 'MAGE Score':[mage], "Excursions":[excursions],'Days':[day], 'Daytime Mean:':[mean_day], 'Nighttime Mean':[mean_night]})

        data_description = data_description.set_index(['Subject ID'], drop=True)

        display(data_description)


#==================================================================================================================
#   GVI's
#==================================================================================================================

    def gvIndices(self, data = consolidatedData):
        data_description = pd.DataFrame()
        for subjectId, df in data.groupby('subjectId'):
        #     print(subjectId)
            df = self.fullDaysOnly(df)
            df['Display Time'] = pd.to_datetime(df['Display Time'])
            df=df.reset_index(drop=True)

            dates = []
            for i in range(len(df.index)):
                dates.append(df['Display Time'][i].date())
            df['Date'] = dates
            
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
            
            start = df.Date[0]
            
            end = start+timedelta(days=7)
            
            last_date = df.Date.iloc[-1]
            
            pgs_weekly = []
            while end <= last_date:
                xy = pd.DataFrame()
                mask = (df['Date'] >= start) & (df['Date'] <= end)
                xy = pd.concat([xy, df.loc[mask]],ignore_index=True)
                pgs_weekly.append(self.pgs(xy, units='mg'))
                start = end+timedelta(days=1)
                end = start+timedelta(days=7)  

            xy = pd.DataFrame()    
            end = last_date
            mask = (df['Date'] >= start) & (df['Date'] <= end)
            xy = pd.concat([xy, df.loc[mask]],ignore_index=True)
            pgs_weekly.append(self.pgs(xy, units='mg'))
            
            pgs_value = mean(pgs_weekly)
            
            dt = self.dt(df)
            
            TAR_VH, TAR_H, TIR, TBR_L, TBR_VL = self.tir(df, units='mg')

            xx = self.subSample(df)
            hypo, hyper = self.variabilityEpisodes(xx, "mg")

            igc, hypoglycemicIndex, hyperglycemicIndex = self.IGC(df, 'mg')

            li = self.glucoseLiabilityIndex(df, 'mg')

            adrr_val = self.adrr(df, 'mg')

            modd_val = self.modd(df)

            temp_df = pd.DataFrame({'Subject ID':[subjectId], 'GFI':[round(gfi,3)], 'GCF':[round(gcf,3)], 'LBGI':[round(LBGI,3)], 'HBGI':[round(HBGI,3)], 'BGRI':[round(BGRI,3)], 'GRADE':[round(GRADE,3)], 'HypoG_P':[round(HypoG_P,3)],'EuG_P':[round(EuG_P,3)], 'HyperG_P':[round(HyperG_P,3)], 'J Index':[round(j_index,3)], 'Mvalue':[round(Mvalue,3)], 'MAG':[round(MAG,3)], 'GVP':[round(GVP,3)], 'GMI':[round(GMI,3)], 'LAGE':[round(LAGE,3)],'MAX':[round(MAX,3)], 'MIN':[round(MIN,3)], 'HBA1C':[round(HBA1C,3)], 'MEAN':[round(m,3)], 'STD-DEV':[round(sd,3)],'CV':[round(cv,3)], 'IQR':[round(iqr,3)], 'SDRC':[round(sdrc,3)], 'PGS':[round(pgs_value,3)], 'DT':[round(dt,3)], 'TAR_VH(%)': [round(TAR_VH,3)], 'TAR_H(%)': [round(TAR_H,3)], 'TIR(%)': [round(TIR,3)], 'TBR_L(%)': [round(TBR_L,3)], 'TBR_VL(%)': [round(TBR_VL,3)], 'Hypoglycemic Episodes': [hypo], 'Hyperglycemic Episodes': [hyper], "IGC": [igc], "Hypoglycemic Index": [hypoglycemicIndex], "Hyperglycemic Index": [hyperglycemicIndex], "Liability Index": [li], "ADDR": [adrr_val], "MODD": [modd_val]})

            data_description = pd.concat([data_description,temp_df],ignore_index=True)

        # data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)

        display(data_description)
        
        data_description.to_csv(self.cwd+"/TSForecasting/Data/Glucose Indices.csv")

    def individualGvIndices(self, uid, data = consolidatedData):
        df = data[data['subjectId']==str(uid)]
        df = self.fullDaysOnly(df)
        df['Display Time'] = pd.to_datetime(df['Display Time'])
        df=df.reset_index(drop=True)

        dates = []
        for i in range(len(df.index)):
            dates.append(df['Display Time'][i].date())
        df['Date'] = dates   
        
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
        
        start = df.Date[0]
        end = start+timedelta(days=7)
        last_date = df.Date.iloc[-1]
        pgs_weekly = []
        while end <= last_date:
            xy = pd.DataFrame()
            mask = (df['Date'] >= start) & (df['Date'] <= end)
            xy = pd.concat([xy, df.loc[mask]],ignore_index=True)
            pgs_weekly.append(self.pgs(xy, units='mg'))
            start = end+timedelta(days=1)
            end = start+timedelta(days=7)  

        xy = pd.DataFrame()    
        end = last_date
        mask = (df['Date'] >= start) & (df['Date'] <= end)
        xy = pd.concat([xy, df.loc[mask]],ignore_index=True)
        pgs_weekly.append(self.pgs(xy, units='mg'))
        
        pgs_value = mean(pgs_weekly)

        dt = self.dt(df)
        
        TAR_VH, TAR_H, TIR, TBR_L, TBR_VL = self.tir(df, units='mg')

        xx = self.subSample(df)
        hypo, hyper = self.variabilityEpisodes(xx, "mg")

        igc, hypoglycemicIndex, hyperglycemicIndex = self.IGC(df, 'mg')

        li = self.glucoseLiabilityIndex(df, 'mg')

        adrr_val = self.adrr(df, 'mg')

        modd_val = self.modd(df)

        data_description = pd.DataFrame({'Subject ID':[uid], 'GFI':[round(gfi,3)], 'GCF':[round(gcf,3)], 'LBGI':[round(LBGI,3)], 'HBGI':[round(HBGI,3)], 'BGRI':[round(BGRI,3)], 'GRADE':[round(GRADE,3)], 'HypoG_P':[round(HypoG_P,3)],'EuG_P':[round(EuG_P,3)], 'HyperG_P':[round(HyperG_P,3)], 'J Index':[round(j_index,3)], 'Mvalue':[round(Mvalue,3)], 'MAG':[round(MAG,3)], 'GVP':[round(GVP,3)], 'GMI':[round(GMI,3)], 'LAGE':[round(LAGE,3)],'MAX':[round(MAX,3)], 'MIN':[round(MIN,3)], 'HBA1C':[round(HBA1C,3)], 'MEAN':[round(m,3)], 'STD-DEV':[round(sd,3)],'CV':[round(cv,3)], 'IQR':[round(iqr,3)], 'SDRC':[round(sdrc,3)], 'PGS':[round(pgs_value,3)], 'DT':[round(dt,3)], 'TAR_VH(%)': [round(TAR_VH,3)], 'TAR_H(%)': [round(TAR_H,3)], 'TIR(%)': [round(TIR,3)], 'TBR_L(%)': [round(TBR_L,3)], 'TBR_VL(%)': [round(TBR_VL,3)], 'Hypoglycemic Episodes': [hypo], 'Hyperglycemic Episodes': [hyper], "IGC": [igc], "Hypoglycemic Index": [hypoglycemicIndex], "Hyperglycemic Index": [hyperglycemicIndex], "Liability Index": [li], "ADDR": [adrr_val], "MODD": [modd_val]})


        # data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)

        display(data_description)


    def gfi(self, x):
        """
            Glucose Fluctuation Index and Glucose Coefficient of Fluctuation
            The GFI is based on consecutive glucose differences, where consecutive differences
            in GFI are squared prior to finding their mean and taking the square root.
            The potential benefit is that differences are weighted individually, giving more 
            importance to the greatest ones, which are likely to be more detrimental 

            GCF is computed as the ratio of GFI to the mean of input glucose values.

            DESCRIPTION: Function takes in a sequesnce of continuous glucose values, 
            and computes glucose fluctuation index (GFI) 
            and the glucose coefficient of fluctuation (GCF).
            This function accepts data given either in mmol/L or mg/dl.

            FUNCTION PARAMETERS:  x  - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings;

            RETURN: Output is Pandas dataframe that contains numeric values for GFI and GCF accordingly;

            REFERENCES:
            - Le Floch J, Kessler L (2016). “Glucose variability: comparison of 
            different indices during continuous glucose monitoring in diabetic patients.” 
            Journal of diabetes science and technology, 10(4), 885–891.
        """
        N = len(x)
        S = 0
        for i in range(0,N-1):
            S = S + (x.iloc[i, 1]  - x.iloc[(i+1), 1]) ** 2
            
        gfi = np.sqrt(S/N)
        gcf = gfi/np.mean(x.iloc[:,1])
        # return pd.DataFrame({'GFI':[gfi], 'GCF':[gcf]})
        if math.isinf(gfi):
            print("Error calculating GFI for: "+str(x["subjectId"]))
            gfi = 0.0
        elif math.isinf(gcf):
            print("Error calculating GCF for: "+str(x["subjectId"]))
            gcf = 0.0

        return gfi, gcf


    def bgri(self, x, units):
        """
            Blood Glucose Risk Index
            LBGI is a measure of the frequency and extent of low blood glucose (BG) readings;
            HBGI is a measure of the frequency and extent of high BG readings;
            BGRI is a measure for the overall risk of extreme BG equal to LBGI + HBGI.

            The LBGI has been validated as a predictor of severe hypoglycemia, while the HBGI has 
            been related to risk for hyperglycemia and HbA1c;
            Both indices demonstrate high sensitivity to changes in glycemic profiles and metabolic 
            control, as well as high sensitivity to the effects of treatment. 

            Larger values of LBGI and HBGI indicate higher risk for hypoglycemia and hyperglycemia, 
            respectively.
            Although originally derived using self-monitored blood glucose data, these parameters 
            have been adapted to continuous interstitial glucose monitoring data.
            Correlations between LBGI and subsequent hypoglycemia and between HBGI and HbA1c have 
            been reported.

            The LBGI and the HBGI are non-negative numbers; each index and their sum could range 
            theoretically between 0 and 100.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes:
            low blood glucose index (LBGI), high blood glucose index (HBGI),
            and overall blood glucose risk index (BGRI).
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x  - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings;
            units -  should be set either to "mmol" or to "mg";

            RETURN: Output is Pandas dataframe that contains numeric values for LBGI, HBGI and BGRI accordingly;
            details  LBGI is a measure of the frequency and extent of low blood glucose (BG) readings;

            REFERENCES:
            - Service FJ (2013). “Glucose variability.” Diabetes, 62(5), 1398.
            - Kovatchev BP, Clarke WL, Breton M, Brayman K, McCall A (2005).
            “Quantifying temporal glucose variability in diabetes via continuous
            glucose monitoring: mathematical methods and clinical application.”
            Diabetes technology & therapeutics, 7(6), 849–862.
        """
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
        if math.isinf(LBGI):
            print("Error calculating LBGI for: "+str(x["subjectId"]))
            LBGI = 0.0
        elif math.isinf(HBGI):
            print("Error calculating HBGI for: "+str(x["subjectId"]))
            HBGI = 0.0
        elif math.isinf(BGRI):
            print("Error calculating BGRI for: "+str(x["subjectId"]))
            BGRI = 0.0

        return LBGI, HBGI, BGRI


    def grade(self, x, units):
        """
            Glycaemic Risk Assessment Diabetes Equation
            GRADE is a score derived to summarize the degree of risk associated with a certain glucose profile.
            Qualitative risk scoring for a wide range of glucose levels inclusive of marked hypoglycemia 
            and hyperglycemia is obtained based on a committee of diabetes practitioners.
            The calculated score can range from 0 -- meaning no risk to 50 -- meaning maximal risk.
            The structure of the formula is designed to give a continuous curvilinear approximation with a nadir at 
            4.96 mmol/L (90 mg/dL) and high adverse weighting for both hyper- and hypoglycaemia.
            The contribution of hypoglycaemia, euglycaemia and hyperglycaemia to the GRADE score is expressed as 
            percentages:  e.g.  GRADE  (hypoglycaemia%, euglycaemia%, hyperglycaemia%),
            which are defined as:

             <3.9 mmol/L (70 mg/dL) hypoglycaemia;

             3.9 - 7.8mmol/L (70–140 mg/dL) euglycemia;

             and >7.8 mml/L (140 mg/dL) hyperglycemia.


            DESCRIPTION: Takes in a sequesnce of continuous glucose values 
            and computes Glycaemic Risk Assessment Diabetes Equation (GRADE) score.
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings;
            units -  should be set either to "mmol" or to "mg";

            RETURN: Output is Pandas dataframe with numeric values for GRADE and percentages expressing risk calculated 
            from hypoglycaemia, euglycaemia and hyperglycaemia;

            REFERENCES:
            - Service FJ (2013). “Glucose variability.” Diabetes, 62(5), 1398.
            - Hill N, Hindmarsh P, Stevens R, Stratton I, Levy J, Matthews D (2007). 
            “A method for assessing quality of control from glucose profiles.” 
            Diabetic medicine, 24(7), 753–758.

        """
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
        if math.isinf(GRADE):
            print("Error calculating GRADE for: "+str(x["subjectId"]))
            GRADE = 0.0
        elif math.isinf(HypoG_P):
            print("Error calculating HypoG_P for: "+str(x["subjectId"]))
            HypoG_P = 0.0
        elif math.isinf(EuG_P):
            print("Error calculating EuG_P for: "+str(x["subjectId"]))
            EuG_P = 0.0
        elif math.isinf(HyperG_P):
            print("Error calculating HyperG_P for: "+str(x["subjectId"]))
            HyperG_P = 0.0

        return GRADE , HypoG_P, EuG_P, HyperG_P

    
    def j_index(self, x, units):
        """
            J-index
            The J-index definition includes a sandard deviation into the measurement of glycemic variability.
            This index was developed to stress the importance of the two major glycemia components: mean level 
            and variability.
            J-index can be used to describe glucose control using the following scheme:
             - Ideal glucose control 10 ≤ J-index ≤ 20;
             - Good glucose control 20 < J-index ≤ 30;
             - Poor glucose control 30 < J-index ≤ 40;
             - Lack of glucose control J-index > 40.
            Originally derived from intermittent blood glucose determinations,
            it has been adapted to continuous monitoring data too.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes J-index.
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings;
            units -  should be set either to "mmol" or to "mg";

            RETURN: Output is Pandas dataframe that contains numeric value for J-index;

            REFERENCES:
             - Wojcicki J (1995). “J-index. A new proposition of the assessment of current glucose 
            control in diabetic patients.” Hormone and metabolic research, 27(01), 41–42.
             - Service FJ (2013). “Glucose variability.” Diabetes, 62(5), 1398.

        """
        if (units == 'mg'):
            a = 0.001
        elif (units=='mmol'):
            a = 0.324
        else:
            print('units should be either mmol or mg')
            return 0
        
        j = a*(np.mean(x.iloc[:, 1]) + np.std(x.iloc[:, 1])) ** 2
        
        # return pd.DataFrame({'J-index':[j]})
        if math.isinf(j):
            print("Error calculating J-Index for: "+str(x["subjectId"]))
            j = 0.0
        return j


    def m_value(self, x, units, ref_value):
        """
            M-value
            Originally, M-value was defined as a quantitative index of the lack of efficacy of the treatment in 
            the individual diabetic patient.
            Othervise, M-value was proposed as a result of trying to quantify the glycemic control of diabetes patients.
            It is a measure of the stability of the glucose excursions in comparison with an “ideal” 
            glucose value of 120 mg/dL; developed using six self-monitored blood glucose values over 
            24 h in 20 patients with type-I diabetes.
            In the final M-value exression, choice of the ideal glucose/reference value is left for the user.
            The M-value is zero in healthy persons, rising with increasing glycemic variability or poorer 
            glycemic control.
            M-value, should be calculated for each individual day (i.e. over 24h).
            The formula gives greater emphasis to hypoglycemia than hyperglycemia, making it difficult 
            to distinguish between patients with either high mean glucose or high glucose variability.
            Thus the M-value is not an indicator solely of glucose variability but is a hybrid measure of
            both variability and mean glycemia.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes M-value.
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings taken e.g. over one day (24h);
            units -  should be set either to "mmol" or to "mg";
            ref_value  - gives option to set a reference value; e.g. use 120mg/dL to reflect original M-value formula,
            set to 80mg/dL for whole blood, set to 90mg/dL for plasma measurements of glucose;

            RETURN: Output is Pandas dataframe that contains numeric value for M-value.

            REFERENCES:

            - Schlichtkrull J, Munck O, Jersild M (1965). “The M-value, an index of blood-sugar control in 
            diabetics.” Acta Medica Scandinavica, 177(1), 95–102.
            - Service FJ (2013). “Glucose variability.” Diabetes, 62(5), 1398.
            - Siegelaar SE, Holleman F, Hoekstra JB, DeVries JH (2010). “Glucose variability; does it matter?” 
            Endocrine reviews, 31(2), 171–182.
        """
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
        if math.isinf(Mvalue):
            print("Error calculating Mvalue for: "+str(x["subjectId"]))
            Mvalue = 0.0

        return Mvalue


    def mag(self, x):
        """
            Mean Absolute Glucose Change
            The MAG is based on added-up differences between sequential blood glucose profiles
            per 24h divided by the time in hours between the first and last blood glucose measurement.
            It measures both the amplitude and frequency of oscillations.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            mean absolute glucose change (MAG).
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for MAG.

             REFERENCES:
            - Hermanides J, Vriesendorp TM, Bosman RJ, Zandstra DF, Hoekstra JB, DeVries JH (2010). 
            “Glucose variability is associated with intensive care unit mortality.” 
            Critical care medicine, 38(3), 838–842.
        """
        S = np.abs(np.sum(x.iloc[:, 1].diff()))
        n = len(x)-1
        total_T = (x.iloc[n,0] - x.iloc[0, 0])/np.timedelta64(1,'h')
        MAG = S/total_T
        # return pd.DataFrame({'MAG':[MAG]})
        
        if math.isinf(MAG):
            print("Error calculating MAG for: "+str(x["subjectId"]))
            MAG = 0.0

        return MAG

    
    def gvp(self, x, units):
        """
        Glycemic Variability Percentage
        GVP can provide a quantitative measurement of glycemic variability over a given interval of 
        time by analyzing the length of the CGM temporal trace normalized to the duration under evaluation.
        It is expressed as a percentage above the minimum line length with zero glycemic variability.
        This metric gives equal weight to both the amplitude and frequency.
        GVP value does contain a dependency on the unit of measure of glucose (mg/dL or mmol/L)
        It is recommended to perform calculation in glucose units of mg/dL.
        Recommended sampling intervals should not exeede 15min, greater sampling intervals such as 30 or 60 min 
        are not suitable for use with the GVP metric.
        This method is best suited for CGM traces with high data recording rate and a low number of data omissions. 

        DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
        glycemic variability percentage (GVP).
        This function accepts data given mg/dL only.

        FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
        in the second - Pandas time stamp, and in the fird - numeric values of 
        continuous glucose readings.

        RETRUN: Output is Pandas dataframe that contains numeric value for GVP.

        REFERENCES:
        - T. A. Peyser, A. K. Balo, B. A. Buckingham, I. B. Hirsch, and A. Garcia. Glycemic variability percentage:
        a novel method for assessing glycemic variability from continuous glucose monitor data. 
        Diabetes technology & therapeutics, 20(1):6–16, 2018.
   
        """
        if (units != 'mg'):
            print('units can only be mg')
            return 0
        
        dt = x.iloc[:, 0].diff()/np.timedelta64(1,'m') # assuming that sampling can not necessarily be equally spaced
        dy = x.iloc[:, 1].diff()
        
        L = np.sum(np.sqrt(dt**2 + dy**2))
        L_0 = np.sum(dt)
        
        GVP = (L/L_0 -1) *100
        # return pd.DataFrame({'GVP(%)':[GVP]})

        if math.isinf(GVP):
            print("Error calculating GVP for: "+str(x["subjectId"]))
            GVP = 0.0

        return GVP


    def gmi(self, x, units):
        """
            Glucose Management Indicator
            GMI is calculated from a formula derived from the regression line computed from a plot 
            of mean glucose concentration points on the x-axis and contemporaneously measured A1C values 
            on the y-axis ( replacement to "estimated A1C"). It was rerived using a Dexcom sensor, threfore there is no guarantee that 
            this formula would be precisely the same for CGM data collected from a different sensor. 
            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            glycemic variability percentage (GVP).
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for GMI.

            REFERENCES:
            - R. M. Bergenstal, R. W. Beck, K. L. Close, G. Grunberger, D. B. Sacks,A. Kowalski, A. S. Brown, 
            L. Heinemann, G. Aleppo, D. B. Ryan, et al. Glucosemanagement indicator (gmi): a new term for 
            estimating a1c from continuousglucose monitoring. Diabetes care, 41(11):2275–2280, 2018

        """
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
        """
            Largest Amplitude of Glycemic Excursions
            LAGE is the difference between the maximum and minimum glucose values within a day, 
            It is equivalent to a range in statistics and represents the single, biggest fluctuation 
            in glucose level within a day.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            glycemic variability percentage (GVP).
            This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for LAGE, MIN, MAX.

            REFERENCES:
            - TA. Soliman, V. De Sanctis, M. Yassin, and R. Elalaily. Therapeutic use anddiagnostic potential 
            of continuous glucose monitoring systems (cgms) inadolescents.Adv Diabetes Metab, 2:21–30, 2014.
             - M. Tao, J. Zhou, J. Zhu, W. Lu, and W. Jia. Continuous glucose monitoringreveals abnormal features 
            of postprandial glycemic excursions in women withpolycystic ovarian syndrome. 
            Postgraduate medicine, 123(2):185–190, 2011

        """
        MIN = np.min(x.iloc[:, 1])
        MAX = np.max(x.iloc[:, 1])
        LAGE = MAX - MIN
        # return pd.DataFrame({'LAGE': [LAGE], 'MAX': [MAX], 'MIN':[MIN]})
        return LAGE, MAX, MIN


    def ehba1c(self, x):
        """
            Estimated HbA1c 
            Original formula is based on computing estimated glucose level using HbA1c:
            eAC = 28.7*HbA1c - 46.7. Rearranging arguments we can compute eHbA1c.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            glycemic variability percentage (GVP).
            This function works with data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for eHbA1c.

            REFERENCES:
            - G. Bozkaya, E. Ozgu, and B. Karaca. The association between estimated averageglucose
            levels and fasting plasma glucose levels.Clinics, 65(11):1077–1080, 2010
            - https://professional.diabetes.org/diapro/glucose_calc
        """
        HBA1C = (np.mean(x.iloc[:, 1]) + 46.7)/28.7
        # return pd.DataFrame({'eHbA1c': [HBA1C]})

        if math.isinf(HBA1C):
            print("Error calculating HBA1C for: "+str(x["subjectId"]))
            HBA1C = 0.0

        return HBA1C


    def sumstats(self, x):
        """
            Summary Statistics
            Produce a simple summary statistics: mean, standard deviation, coefficient of variation
            and interquartile range.
            
            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            summary statistics: mean, standard deviation, coefficient of variation
            and interquartile range.
            This function works with data given either in mmol/L or mg/dL.
            
            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.
            
            RETRUN: Output is Pandas dataframe that contains numeric value for mean, standard deviation,
            coefficient of variation and interquartile range.
            
        """
        m = np.mean(x.iloc[:, 1])
        sd = np.std(x.iloc[:, 1])
        cv = sd/m
        q75, q25 = np.percentile(x.iloc[:, 1], [75 ,25])
        iqr = q75 - q25
        
        # return pd.DataFrame({'Mean': [m], 'SD':[sd], 'CV': [cv], 'IQR': [iqr]})
        return m, sd, cv, iqr


    def rc(self, x):
        """
            Standard Deviation of the Glucose Rate of Change 
            Glucose rate of change is a way to evaluate the dynamics of glucose fluctuations
            on the time scale of minutes. A larger variation of the glucose rate of change indicates 
            rapid and more pronounced BG fluctuations

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            glycemic variability percentage SDRC.
            Operated on data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for SDRC.

            REFERENCES:
            - W. Clarke and B. Kovatchev. Statistical tools to analyze continuous glucosemonitor data.
            Diabetes technology & therapeutics, 11(S1):S–45, 2009.

        """
        dt = x.iloc[:, 0].diff()/np.timedelta64(1,'m') 
        dy = x.iloc[:, 1].diff()
        
        sdrc = np.std(dy/dt)
        # return pd.DataFrame({'SD of RC': [sdrc]})
       
        if math.isinf(sdrc):
            print("Error calculating SDRC for: "+str(x["subjectId"]))
            sdrc = 0.0

        return sdrc

    
    def pgs(self, x, units):
        """
            Personal Glycemic State
            The PGS is an additive composite metric calculated using the following simple equation
            PGS  = F(GVP) + F(MG) + F(PTIR) + F(H),
            where F(GVP) is a function of the glycemic variability percentage, 
            F(MG) is a function of the mean glucose, 
            F(PTIR) is a function of the percent time in range (from 70 to 180 mg/ dL), and 
            F(H) is a function of the incidence of the number of hypoglycemic episodes per week.
            The hypoglycemia function incorporates two distinct thresholds (54 and 70 mg/dL) and is 
            the sum of two terms: F54(H) and F70(H).
            PGS is computed per week and then averaged across all weeks.
            The min value of the PGS metric is 4.6 corresponding to excellent glycemic control 
            (no diabetes or patients with diabetes under superb glycemic control). 
            The max value of the PGS metric is 40 corresponding to a poor quality of glycemic control 
            that would be seen in patients with elevated A1c values, high mean glucose, and low percent of time 
            in the euglycemic range.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            glycemic variability percentage SDRC.
            Operated on data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for DT.

            REFERENCES:
            -  I. B. Hirsch, A. K. Balo, K. Sayer, A. Garcia, B. A. Buckingham, and T. A.Peyser. 
            A simple composite metric for the assessment of glycemic status fromcontinuous glucose 
            monitoring data: implications for clinical practice and theartificial pancreas. 
            Diabetes technology & therapeutics, 19(S3):S–38, 2017.

        """
        if (units != 'mg'):
            return print('units can only be mg')
        
        N54 = len(x[x.iloc[:,1]<=54])
        F_54H = 0.5 + 4.5 * (1 - np.exp(-0.81093*N54))
        
        N70 = len(x[x.iloc[:,1]<70]) - N54
        
        if (N70 <= 7.65):
            F_70H = 0.5714 * N70 + 0.625
        else:
            F_70H = 5
            
        F_H = F_54H + F_70H
        GVP = self.gvp(x, units=units)

        F_GVP = 1 + 9/(1 + np.exp(-0.049*(GVP-65.47)))
        
        if len(x)==0:
            lx=1
        else:
            lx = len(x)
        TIR  =  len(x) - len(x[x.iloc[:,1]<70].iloc[:,1]) - len(x[x.iloc[:,1]>180].iloc[:,1])
        PTIR = TIR*100/lx
        
        F_PTIR = 1 + 9/(1 + np.exp(0.0833*(PTIR - 55.04)))
        
        MG = np.mean(x.iloc[:, 1])
        F_MG = 1 + 9 * ( 1/(1 + np.exp(0.1139*(MG-72.08))) + 1/(1 + np.exp(-0.09195*(MG-157.57))) )
        
        PGS = F_GVP + F_MG + F_PTIR + F_H
        # PGS.columns=['PGS']

        if math.isinf(PGS):
            print("Error calculating PGS for: "+str(x["subjectId"]))
            PGS = 0.0

        return PGS


    def dt(self,x):
        """
            The distance traveled 
            This metric is the  sum of the absolute difference in glucose levels for one day
            of consecutive CGM readings. It does not directly calculate frequency or magnitude (amplitude) of excursions; 
            instead, it quantifies the total change in blood glucose levels throughout the day by measuring 
            the total distance from point to point in a daily CGM plot. 
            Thus the greater the distance traveled, the greater the variability.
            DT is computed for each day and then averaged across all days.

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            distance traveled.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID, 
            in the second - Pandas time stamp, and in the fird - numeric values of 
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for  average DT.

            REFERENCES:
            -   D. Rodbard. Glucose variability: a review of clinical applications and research developments.
            Diabetes technology & therapeutics, 20(S2):S2–5, 2018.

        """
        dy = np.sum(np.abs(x.iloc[:, 1].diff()))
        return dy
        # return pd.DataFrame({'DT': [dy]})


    def tir(self, x, units):
        """
            Time in Ranges
            The persentage of time spent witihn the glucoses target range:
            Very high: Time above range (TAR): % of readings and time > 250 mg/dL (>13.9 mmol/L)
            High: Time above range (TAR): % of readings and time 181–250 mg/dL (10.1–13.9 mmol/L)
            In range: Time in range (TIR): % of readings and time 70–180 mg/dL (3.9–10.0 mmol/L)
            Low: Time below range (TBR): % of readings and time 54–69 mg/dL (3.0–3.8 mmol/L)
            Very low: Time below range (TBR): % of readings and time <54 mg/dL (<3.0 mmol/L)

            DESCRIPTION: Takes in a sequesnce of continuous glucose values and computes
            TIR(VH), TIR(H), TAR, TBR(L), TBR(VL).
            This function works with data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS: x - is Pandas dataframe, in the fist column is given subject ID,
            in the second - Pandas time stamp, and in the fird - numeric values of
            continuous glucose readings.

            RETRUN: Output is Pandas dataframe that contains numeric value for  average DT.

            REFERENCES:
            -   T. Battelino, T. Danne, R. M. Bergenstal, S. A. Amiel, R. Beck, T. Biester,E. Bosi,
            B. A. Buckingham, W. T. Cefalu, K. L. Close, et al. Clinical targets forcontinuous glucose monitoring
            data interpretation: recommendations from theinternational consensus on time in range.Diabetes Care,
            42(8):1593–1603, 2019.
        """
        if (units == 'mg'):
            N = len(x)
            TAR_VH = len(x[x.iloc[:,1]> 250])/N*100
            TAR_H  = len(x[(x.iloc[:,1]>= 181) & (x.iloc[:,1]<= 250)])/N*100
            TIR    = len(x[(x.iloc[:,1]>= 70) & (x.iloc[:,1]<= 180)])/N*100
            TBR_L  = len(x[(x.iloc[:,1]>= 54) & (x.iloc[:,1]<= 69)])/N*100
            TBR_VL = len(x[x.iloc[:,1]< 54])/N*100
            # return pd.DataFrame({'TAR_VH(%)': [TAR_VH], 'TAR_H(%)': [TAR_H], 'TIR(%)': [TIR], 'TBR_L(%)': [TBR_L], 'TBR_VL(%)': [TBR_VL]})
            return TAR_VH, TAR_H, TIR, TBR_L, TBR_VL
        elif (units=='mmol'):
            N = len(x)
            TAR_VH = len(x[x.iloc[:,1]> 13.9])/N*100
            TAR_H  = len(x[(x.iloc[:,1]>= 10.1) & (x.iloc[:,1]<= 13.9)])/N*100
            TIR    = len(x[(x.iloc[:,1]>= 3.9) & (x.iloc[:,1]<= 10.0)])/N*100
            TBR_L  = len(x[(x.iloc[:,1]>= 3.0) & (x.iloc[:,1]<= 3.8)])/N*100
            TBR_VL = len(x[x.iloc[:,1]< 3.0])/N*100
            # return pd.DataFrame({'TAR_VH(%)': [TAR_VH], 'TAR_H(%)': [TAR_H], 'TIR(%)': [TIR], 'TBR_L(%)': [TBR_L], 'TBR_VL(%)': [TBR_VL]}
            return TAR_VH, TAR_H, TIR, TBR_L, TBR_VL
        else:
            return print('units should be either mmol or mg')


    def variabilityEpisodes(self, df, unit):
        time_diff = timedelta(hours=0, minutes=15, seconds=30)
        
        
        if unit == 'mg':
            hypoglycemia = df[df.GlucoseValue<=54]
            hyperglycemia = df[df.GlucoseValue>=250]
        elif unit == 'mmol':
            hypoglycemia = df[df.GlucoseValue<=3]
            hyperglycemia = df[df.GlucoseValue>=13.9]
        else:
            print("Unit should be 'mg' or 'mmol'")
            return 0,0
         
        hypoglycemia = hypoglycemia.reset_index(drop=True)
        hypoglycemia['Display Time'] = pd.to_datetime(hypoglycemia['Display Time'])
        hypoglycemia['time_gap'] = hypoglycemia['Display Time'].diff()
        hypoglycemic_episodes = 0
        
        for gap in hypoglycemia['time_gap']:
            if gap <= time_diff:
                hypoglycemic_episodes+=1

        
        
        hyperglycemia = hyperglycemia.reset_index(drop=True)
        hyperglycemia['Display Time'] = pd.to_datetime(hyperglycemia['Display Time'])
        hyperglycemia['time_gap'] = hyperglycemia['Display Time'].diff()
        
        hyperglycemia_episodes = 0
        for gap in hyperglycemia['time_gap']:
            if gap <= time_diff:
                hyperglycemia_episodes+=1
                
        return hypoglycemic_episodes, hyperglycemia_episodes

            # “Number of hypoglycemic (glucose concentrations of less than 3.0 mmol/L (54 mg/dL))and number of hyperglycemic (glucose concentration of more than 13.9 mmol/L (250 mg/dL)) episodes that last at least 15min.
            #  HypoE and HyperE are computed for each day and then averaged across all days within individual time series.

            #  DESCRIPTION: Takes in a sequence of continuous glucose values and computes
            #  HypoE and HyperE.
            #  This function works with data given either in mmol/L or mg/dL.
             
            #  FUNCTION PARAMETERS: x - is Pandas dataframe, in the first column is given subject ID, 
            #  in the second - Pandas timestamp, and in the third- numeric values of 
            #  continuous glucose readings.

            #  RETURN: Output is Pandas dataframe that contains numeric value for average HyperE and HypoE.

            #  REFERENCES:
            # -   T. Battelino, T. Danne, R. M. Bergenstal, S. A. Amiel, R. Beck, T. Biester,E. Bosi,
            #  B. A. Buckingham, W. T. Cefalu, K. L. Close, et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range.Diabetes Care,  42(8):1593–1603, 2019.
            # Code up function to extract Mean24h, day, night - use the function that you already coded up for time stamp.
            # For PGS - lookup for individuals that have at least one week worth of data to test it out, pick those that have least missing values.
    
    
    def IGC(self, df, unit, lltr = 80, ultr = 140, a = 1.1, b = 2.0, c = 30, d = 30):
        if unit == 'mg':
            gv = df['GlucoseValue']
        elif unit == 'mmol':
            gv = 18*df['GlucoseValue']
        else:
            print('Unit should either be mg or mmol')
            return 0
        
        lower_gv = gv[gv < 90]
        upper_gv = gv[gv > 140]
        
        
        count_lower = len(lower_gv.index)
        count_upper = len(upper_gv.index)
        
        hypoglycemicIndex = np.sum(np.power((lltr - lower_gv), b)) / (count_lower*d)   
        hyperglycemicIndex = np.sum(np.power((upper_gv - ultr), a)) / (count_upper*c)
        
        if np.isnan(hypoglycemicIndex):
            hypoglycemicIndex = 0
        if np.isnan(hyperglycemicIndex):
            hyperglycemicIndex=0
        
        igc = hypoglycemicIndex + hyperglycemicIndex
        return round(igc,3), round(hypoglycemicIndex,3), round(hyperglycemicIndex,3)


    def glucoseLiabilityIndex(self,data, unit):
        data = self.hourlySamples(data)
        if unit == 'mg':
            data['GlucoseValue'] = data['GlucoseValue']/18
        gli = np.sum(np.power(data['GlucoseValue'][i] - data['GlucoseValue'][i+1],2) for i in range(0, len(data.index)-1))
        return round(gli,3)


    def adrr(self, xx, unit):
        
        if unit == 'mg':
            f_bg = 1.509*(np.log(xx['GlucoseValue'])**1.084)-5.381
            xx['F(BG)'] = f_bg
        elif unit == 'mmol':
            f_bg = 1.509*(np.log(xx['GlucoseValue']*18)**1.084)-5.381
            xx['F(BG)'] = f_bg
        else:
            print('Unit should either be mg or mmol')
            return 0
        
        dates = []
        for i in range(len(xx.index)):
            dates.append(xx['Display Time'][i].date())
        xx['Date'] = dates 
        

        for Date, df in xx.groupby('Date'):
            r_BG = 0
            rl_BG = [0]
            rh_BG = [0]
            LR = 0
            HR = 0
            ADDR_daily = []
            for f_BG in df['F(BG)']:
                if f_BG < 0:
                    rl_BG.append(f_BG)
                else:
                    rh_BG.append(f_BG)

            LR = max(rl_BG)
            HR = max(rh_BG)
            ADDR_daily.append(LR+HR)
        
        
        return round(mean(ADDR_daily),3)
        
               
    def modd(self, data):
        data = self.subSample(data)
        data['Display Time'] = data['Display Time'].dt.round('5min') 
        
        times = []
        for i in range(len(data.index)):
            times.append(data['Display Time'][i].time())
        data['Time'] = times  


        Modd = [] 
        s = 0
        gvDiff = 0

        for Time, df in data.groupby('Time'):
            gvDiff = df['GlucoseValue'] - df['GlucoseValue'].shift(-1)
            s = round(gvDiff.sum(),3)
            Modd.append(s)
        return round(mean(Modd),3) 

#==================================================================================================================
#   Helper Methods
#==================================================================================================================
    def meanCalculations(self, xx):
        dates = []
        times = []
        for i in range(len(xx.index)):
            dates.append(xx['Display Time'][i].date())
            times.append(xx['Display Time'][i].time())
        xx['Date'] = dates   
        xx['Time'] = times
        
        n_s = datetime.strptime("00:00:00","%H:%M:%S").time()
        n_e = datetime.strptime("06:00:00","%H:%M:%S").time()
        d_s = datetime.strptime("06:00:01","%H:%M:%S").time()
        d_e = datetime.strptime("23:59:59","%H:%M:%S").time()
        
        day_means = []
        night_means = []
        for Date, df in xx.groupby('Date'):
        #     print(Date)
            days = pd.DataFrame()
            night = pd.DataFrame()
            day_readings = (df['Time'] >= d_s) & (df['Time'] <= d_e)
            night_readings = (df['Time'] >= n_s) & (df['Time'] <= n_e)
            days = df.loc[day_readings]
            night = df.loc[night_readings]
            if (days.empty) == False:
                day_means.append(mean(days.GlucoseValue))
            if (night.empty) == False:
                night_means.append(mean(night.GlucoseValue))
                
        return round(mean(day_means),3), round(mean(night_means),3)
    

    def mageCalculation(self, df, std=1):
        
        #extracting glucose values and incdices
        glucs = df['GlucoseValue'].to_list()
        indices = [1*i for i in range(len(glucs))]
        stdev = std
        
        # detection of local minima and maxima
        x = indices
        gvs = glucs
        # local min & max
        a = np.diff(np.sign(np.diff(gvs))).nonzero()[0] + 1      
        # local min
        valleys = (np.diff(np.sign(np.diff(gvs))) > 0).nonzero()[0] + 1 
        # local max
        peaks = (np.diff(np.sign(np.diff(gvs))) < 0).nonzero()[0] + 1         
        # +1 due to the fact that diff reduces the original index number

        #storing the local minima and maxima to identify and remove turning points
        excursion_points = pd.DataFrame(columns=['Index', 'Timestamp', 'GlucoseValue', 'Type'])
        k=0
        for i in range(len(peaks)):
            excursion_points.loc[k] = [peaks[i]] + [df['Display Time'][k]] + [df['GlucoseValue'][k]] + ["P"]
            k+=1

        for i in range(len(valleys)):
            excursion_points.loc[k] = [valleys[i]] + [df['Display Time'][k]] + [df['GlucoseValue'][k]] + ["V"]
            k+=1

        excursion_points = excursion_points.sort_values(by=['Index'])
        excursion_points = excursion_points.reset_index(drop=True)
        # display(excursion_points)


        # selecting turning points
        turning_points = pd.DataFrame(columns=['Index', 'Timestamp', 'GlucoseValue', 'Type'])
        k=0
        for i in range(stdev,len(excursion_points.Index)-stdev):
            positions = [i-stdev,i,i+stdev]
            for j in range(0,len(positions)-1):
                if(excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]):
                    if(excursion_points.Type[positions[j]]=='P'):
                        if excursion_points.GlucoseValue[positions[j]]>=excursion_points.GlucoseValue[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
                    else:
                        if excursion_points.GlucoseValue[positions[j]]<=excursion_points.GlucoseValue[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j]]
                            k+=1
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1

        if len(turning_points.index)<10:
            turning_points = excursion_points.copy()
            excursion_count = len(excursion_points.index)
        else:
            excursion_count = len(excursion_points.index)/2



        turning_points = turning_points.drop_duplicates(subset= "Index", keep= "first")
        turning_points=turning_points.reset_index(drop=True)
        excursion_points = excursion_points[excursion_points.Index.isin(turning_points.Index) == False]
        excursion_points = excursion_points.reset_index(drop=True)
            # display(turning_points)

        # calculating the MAGE score
        mage = turning_points.GlucoseValue.sum()/excursion_count
        

        return round(mage,3), excursion_count






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
            
            
    def fullDay(self, data):
        """
           trimming an individual's glucose values to only consist of full days     
            Input:
                data: irregular time series
            Output:
                data: time series with data only for full days
        """ 
        
        dates = list()
        data = data.reset_index(drop=True)
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


    def fullDaysOnly(self, data):
        data_fullDays = pd.DataFrame()

        for subjectId, df in data.groupby('subjectId'):
            df['Display Time'] = pd.to_datetime(df['Display Time'])
            df = df.reset_index(drop=True)
            temp = self.fullDay(df)
            data_fullDays = pd.concat([data_fullDays, temp],ignore_index=True)

        return(data_fullDays)


    def hourlySamples(self, df):
        groupkey = df['Display Time'].values.astype('datetime64[h]')
        result = df.groupby(groupkey).mean()
        result = result.reset_index(drop=True)
        return (result)


        
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
 
