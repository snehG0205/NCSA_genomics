# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:37:04 2020

@author: snehgajiwala
"""

# basics
import pandas as pd
import numpy as np
import os
import math
import random
import re

# keras -> deep learning model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

# DCT Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# other operations
from scipy import stats
from statistics import mean

# for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# for datetime operations
from dateutil.parser import parse
from datetime import timedelta
from datetime import datetime

# suppress warnings
import warnings  
warnings.filterwarnings('ignore')





class glucoCheckOps:



#==================================================================================================================
#   Local variables
#==================================================================================================================

    model = None
    model_history = None

    cwd = os.getcwd()

    all_data = pd.read_csv(cwd+'/GlucoCheck/Data/consolidatedDataForPackage.csv')
    metadata = pd.read_csv(cwd+'/GlucoCheck/Data/consolidatedMetadata.csv')
    hallData = pd.read_csv(cwd+'/GlucoCheck/Data/Hall/data_hall_raw.csv')
    gluVarProData = pd.read_csv(cwd+'/GlucoCheck/Data/Gluvarpro/Gluvarpro.csv')
    cgmData = pd.read_csv(cwd+'/GlucoCheck/Data/CGM/CGManalyzer.csv')
    ohioData = pd.read_csv(cwd+'/GlucoCheck/Data/Ohio-Data/OhioFullConsolidated.csv')




#==================================================================================================================
#   Core Methods
#==================================================================================================================

    def __init__(self):
        """
            Package name: GlucoCheck

            Class name: TimeSeriesForecast

            filename: GlucoCheck

            Import code: from GlucoCheck.glucoCheck import glucoCheckOps

            Creating an object: object = glucoCheckOps()

            Package dependencies:
                - pandas
                - numpy
                - matplotlib
                - dateutil
                - datetime
                - re 
                - sklearn
                - keras
                - scipy
                - random
                - os
                - math
                - statistic

            Read complete documentation here:
            https://wiki.ncsa.illinois.edu/display/CPRHD/Package+Documentation
        """

        # print("Object Created!")

    
    def splitIndividual(self, inputdata, missingPerc):
        """
        This method is used to split the individual data with big gap into two part and remove the part with less entries

        Function Parameters:
        data: the dataset (CSV file) entered by the user to convert the timestamp. It should have the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        type: pandas DataFrame

        Return:
        The output is two a pandas data frame split by the max gap

        """
        inputdata['time_gap'] = inputdata['Display Time']- inputdata['Display Time'].shift(1)
        inputdata['gaps'] = inputdata['time_gap'].dt.total_seconds() / (5*60)
        temp_df = inputdata
        temp_df['time_gap'].iloc[0] = pd.NaT
        temp_df['time_gap'].iloc[0] = timedelta(hours=0, minutes=0, seconds=0)

        idx = temp_df['time_gap'].idxmax()
            
        temp_df1 = temp_df.loc[:idx-1]
        # newid1 = str(temp_df1['subjectId'].iloc[1]) + "-1"
        # temp_df1['subjectId'] = newid1

            
        temp_df2 = temp_df.loc[idx:]
        newid2 = str(temp_df1['subjectId'].iloc[1]) + "-2"
        temp_df2['subjectId'] = newid2

        temp_df1 = temp_df1.drop(['time_gap','gaps'], axis=1)
        temp_df1 = temp_df1.reset_index(drop=True)

        temp_df2 = temp_df2.drop(['time_gap','gaps'], axis=1)
        temp_df2 = temp_df2.reset_index(drop=True)
    

        return temp_df1, temp_df2

    
    def gapInfo(self, inputdata):
    
        """
        This method is used to detect gap sizes for each individual dataset
        
        Function Parameters:
        data: the dataset (CSV file) entered by the user to convert the timestamp. It should have the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        type: pandas DataFrame

        Return:
        The output is a panda data frame includes the gap size information for each individual

        """
    
        inputdata['time_gap'] = inputdata['Display Time']- inputdata['Display Time'].shift(1)
        inputdata['gaps'] = inputdata['time_gap'].dt.total_seconds() / (5*60)
        gap_info = pd.DataFrame()

    
        for subjectId, df in inputdata.groupby('subjectId'):
            subj_id = str(subjectId)
            df['time_gap'].iloc[0] = pd.NaT
            totalGapSize = df[df["time_gap"]>str("00:05:10")]
            gaps = totalGapSize['gaps']
            temp_df = pd.DataFrame({'Subject ID':[subj_id], 'gaps': [gaps]})
            gap_info = pd.concat([temp_df,gap_info],ignore_index=True)
        
        return gap_info


    def datePreprocess(self,data):
        """
            This method is used to preprocess the data entered by the user. It identifies the date and converts it to the standard datetime format (%Y-%m-%d %H:%M:%S). It also converts the Timestamp to the index of the data frame

            Function Parameters:
            data :   the dataset (CSV file) entered by the user to convert the timestamp. It should have the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            type: pandas DataFrame

            Return:
            The output is a pandas data frame of the csv file supplied with the 'Display Time' converted to the (%Y-%m-%d %H:%M:%S) format.        
        """

        length = data.shape[0]
        for i in range(0,length):
            #print(i)
            s = str(data.iloc[i]['Display Time'])
            k = re.sub("[^0-9]", "", s)
            datetimeObj = parse(k) 
            data = data.replace(to_replace = s, value = datetimeObj)
        
        data = data.set_index(['Display Time'], drop=True)
        
        return(data)


    def split_sequence(self,sequence, n_steps):
        """
        This method is used to split an individual with a very large ap size dataset
        
        Function Parameters:
        sequence: the data of an individual. It should have the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        type: pandas DataFrame

        n_steps: the length of gaps above which we would like to split the data

        Return:
        The output is two pandas data frames 

        
        """
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    def train(self, data):
        """
            This method is used to train the LSTM imputation model on data. The default data set for the training includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package, and the Ohio University dataset. The user may specify their own data if they wish to. 

            Function Parameters:
            data: the dataset (CSV file) entered by the user to train the model. It should have the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            type: pandas DataFrame

            Return:
            The output is a model trained on the supplied data that can be used to perform imputations imputations
        """ 
        print("Training Model...\n")  
        
        raw_seq = data.GlucoseValue.tolist()
        # choose time steps
        n_steps = 4
        # split into samples
        X, y = self.split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        n_features = 1
        n_seq = 2
        n_steps = 2
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

        # define model
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters=128, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features), go_backwards=True))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        # fit model
        self.model_history = self.model.fit(X, y, epochs=50, verbose=0)
        print("Model trained successfully!")


    def getModelMetrics(self):
        '''
        This method is used to return the model metrics
        
        Function Parameters:

        Return:
        Prints the model metrics
        '''
        print("Model loss on training set:")
        print(self.model_history.history['loss'])
        print("Model Accuracy on training set:")
        print(self.model_history.history['accuracy'])
        print("Model loss on validation set:")
        print(self.model_history.history['val_loss'])
        print("Model accuracy on validation set:")
        print(self.model_history.history['val_accuracy'])
        

    def impute(self, dataWithMissing):
        '''
        This method is used to impute the gaps in an individual
        
        Function Parameters:
        dataWithMissing: the data of an individual with a gap:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        type: pandas DataFrame

        Return:
        The output is a pandas data frame with gap imputed 
        '''

        start,end,gap = self.detectGap(dataWithMissing)

        if gap<=5:
            print("Gap < 5; We use the linear imputations")
            dataWithMissing['GlucoseValue'] = dataWithMissing['GlucoseValue'].interpolate(method='linear', limit_direction='both')
            imputed_data = dataWithMissing
            print("Imputation Successful")
            return imputed_data

        elif gap<=15:
            print("Gap < 15; We use the spline imputations")
            dataWithMissing['GlucoseValue'] = dataWithMissing['GlucoseValue'].interpolate(method='spline', limit_direction='both',order=4)
            imputed_data = dataWithMissing
            print("Imputation Successful")
            return imputed_data

        elif gap<50:
            print("Gap < 50; We use LSTM imputations")
            dataBeforeGap = dataWithMissing[:start]
            dataBeforeGap = self.smoothing(dataBeforeGap)
            self.train(dataBeforeGap);
            imputed_data = self.LSTMimpute(dataWithMissing)
            print("Imputation Successful")
            return imputed_data

        else:
            print("We cannot impute this data")
            return 0


    def LSTMimpute(self,test_data):
        """
            This method performs the imputations on gaps present in the individual's glucose values using the trained LSTM model
            
            Function Parameters:
            test_data: the dataset (CSV file) entered by the user with a gap that needs to be imputed. It should have the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            type: pandas DataFrame
            
            flag:A flag variable to decide whether to return the imputed values as a pandas data frame (flag=1)  or to save the imputed values as a CSV file and plot the graph for it (flag=0). The default value is 0.  type: integer

            Return:
            The imputed values as a pandas data frame (flag=1)  or  the imputed values saved as a csv file and a line graph for it

        """
        
        # test_data = self.datePreprocess(test_data)
        b,e,g = self.detectGap(test_data)
        test = test_data[:b]
        vals = test.GlucoseValue
        preds = []
        N = 4
        x_input = np.array(vals[-N:].tolist())
        n_features = 1
        n_seq = 2
        n_steps = 2
        k=0
        while k<g:
            x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
            # x_input = x_input.reshape((1, n_seq, n_steps, n_features))
            yhat = self.model.predict(x_input, verbose=0)
            x = round(yhat[0][0])
            preds.append(x)
            x_input = np.append(x_input,x)
            x_input = x_input[-N:]
            k+=1
        
        
        # p = pd.Series(preds)
        
        # q = p.ewm(alpha=0.05, adjust=False).mean()
        # q = q.round()
        

        test_data.GlucoseValue[b:e] = preds
        test_data[e:] = self.smoothing(test_data[e:])
        
        return test_data


    def plotIndividual(self, uid, data, date = None):
        """
            This method plots the graph of the Glucose Values of a single Subject ID. The subject ID can be found as a part of the description of the data. The default data includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package, and the Ohio University dataset. The user may specify their own data if they wish to. The subject ID is a part of the data supplied by the user. 

            Function Parameters:

            uid: The subject ID of the user to plot  type: String
            data: the dataset (CSV file) entered by the user to train the model. It should have the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            type: pandas DataFrame

            Return:
            A line graph of the Subject ID's Glucose Value fluctuations with respect to the timestamp

        """
        if date != None:
            print("Displaying for day: "+str(date))
            new = data[data.iloc[:,0]==str(uid)]
            # new = new.astype({'GlucoseValue':int})
            new['GV'] = new.iloc[:,2]
            new['DT'] = pd.to_datetime(new.iloc[:,1])
            new=new.reset_index(drop=True)
            dates = []
            
            for i in range(len(new.index)):
                dates.append(new['Display Time'][i].date())
            
            new['Date'] = dates
            new['Date'] = new['Date'].astype(str)            
            new = new[new['Date']==str(date)]
            # display(new)
            
        else:
            print("Displaying for all days ")
            new = data[data.iloc[:,0]==str(uid)]
            # new['GlucoseValue'] = new.iloc[:,2]
            # new = new.astype({'GlucoseValue':int})
            new['GV'] = new.iloc[:,2]
            new['DT'] = pd.to_datetime(new.iloc[:,1])
            # display(new)

        new = new.reset_index(drop=True)
        new['DT'] = new['DT'].dt.round("5min")
        start_x = new.DT[0]
        end_x = new.DT.iat[-1]
        x_vals = pd.date_range(start=start_x,end=end_x,freq='5T')
        counter = 0
        g = new['GV'].tolist()
        l = new['DT'].tolist()
        y_vals = []
        for x in x_vals:
            if x in l:
                y_vals.append(g[counter])
                counter+=1
            else:
                y_vals.append(float("NaN"))

        df = {'X':x_vals, 'Y':y_vals}
        df = pd.DataFrame(df)

        fig, ax = plt.subplots(figsize=(15, 10))

        plot = sns.lineplot(
            ax=ax,
            data=df, 
            x="X", y="Y",
            hue=df["Y"].isna().cumsum(), 
            palette=["palegreen"]*df["Y"].isna().cumsum().nunique(),
            legend=False, markers=False,
            linewidth = 2
        )
        plot.set_xlabel('Timestamp', weight='bold', fontsize=14)
        plot.set_ylabel('Glucose Value', weight='bold', fontsize=14)
        # ax.set_xticklabels([])

        plt.show()


    def dataDescribe(self, data):
        """
            This method provides a statistical description of the default data used for training the model in the form of a consolidated table. This data has been trimmed to have only complete days with no missing values. This description table is saved as a CSV file for future reference. 

            Function Parameters:
            data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package and the Ohio University with the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            type: pandas DataFrame
            
            meta: The dataset (CSV file) with the metadata about the status of the individuals in the following format:
            ID        object
            status    object
            type: pandas DataFrame

            Return:
            A tabular and graphical representation of the statistical analysis of the consolidated data. This table is also saved as a csv file.
            
        """
        data['Display Time'] = pd.to_datetime(data['Display Time'])
        data['time_gap'] = data['Display Time']- data['Display Time'].shift(1)
        data_description = self.summaryTable(data)
        
        return data_description


    def summaryTable(self, inputdata):
        """
            This method provides a summary table for the basic information of input data.

            Function Parameters:
            data: The default Hall dataset (CSV file) with the following format:
                Display Time     object
                GlucoseValue    float64
                subjectId        object
                type: pandas DataFrame

            Return:
            A consolidated table of all the summary statistics of the input data, including Length of reading, Max Glucose Value, Mean Glucose Value, Missing Values, Percentage of missing values, Average gap size, Days, Start and End.
        """
        data_description = pd.DataFrame()

        for subjectId, df in inputdata.groupby('subjectId'):

            df['time_gap'].iloc[0] = pd.NaT
            subj_id = str(subjectId)
            l_of_r = df['GlucoseValue'].count()
            maxGV = str(df['GlucoseValue'].max())
            minGV = str(df['GlucoseValue'].min())
            meanGV = round(df['GlucoseValue'].mean(),2)

            totalGapSize = df[df["time_gap"]>str("00:05:10")]
            miss_val = round((totalGapSize['time_gap'].sum()).total_seconds() / (60.0*5))

            days = df['Display Time'].iloc[-1]-df['Display Time'].iloc[1]
            s = days.seconds
            h = s//3600
            h = h/24
            float_days = days.days + h
            float_days = round(float_days,2)

            start_time = str(df['Display Time'].iloc[0])
            end_time = str(df['Display Time'].iloc[-1])

            totalEntry = days.total_seconds() / (60.0*5)
            P_miss_val = round(100* miss_val/totalEntry,2)

            df_gap = df[df["time_gap"]>str("00:05:10")]
            if(df_gap.shape[0]==0):
                ave_gap_size = round(miss_val,2)
            else:
                ave_gap_size = round(miss_val / df_gap.shape[0],2)

            idx = df['time_gap'].idxmax()
            max_gap = df['time_gap'].loc[idx]

            temp_df = pd.DataFrame({'Subject ID':[subj_id], 'Start':[start_time], 'End':[end_time], '# of readings':[l_of_r], '# of days':[float_days], 'Timestamp days':[days], '# of missing values':[miss_val], '% of missing values':[P_miss_val], 'Avg gap size':[ave_gap_size], 'Max gv':[maxGV], 'Min gv':[minGV], 'Mean gv':[meanGV]})
            data_description = pd.concat([temp_df,data_description],ignore_index=True)

        return data_description


    def datacleaning(self, inputdata):
        """
            This method performs data cleaning based on the general rule that if there is more than 30% missing value, the max gap size will be inspected and the data will be splited into two part, but only the part with more entries will be kept.

            Function Parameters:
            data: The default Hall dataset (CSV file) with the following format:
                Display Time     object
                GlucoseValue    float64
                subjectId        object
                type: pandas DataFrame

            Return:
            A clean dataset with percentage of missing values below 30%.

        """
        newdf = pd.DataFrame()
        newdf_above50 = pd.DataFrame()

        summary_table = self.summaryTable(inputdata)
        temp_below50 = summary_table.loc[summary_table['Percent of missing values'] < 30]
        temp_above50 = summary_table.loc[summary_table['Percent of missing values'] >= 30]

        id_below50 = list(temp_below50['Subject ID'])
        id_above50 = list(temp_above50['Subject ID'])

        newdf_below50 = inputdata.loc[inputdata['subjectId'].isin(id_below50)]

        for i in id_above50:
            temp_df = inputdata[inputdata['subjectId'] == i]
            temp_df['time_gap'].iloc[0] = pd.NaT
            temp_df['time_gap'].iloc[0] = timedelta(hours=0, minutes=0, seconds=0)

            idx = temp_df['time_gap'].idxmax()

            temp_df1 = temp_df.loc[:idx-1]
            temp_df2 = temp_df.loc[idx:]

            if temp_df1.shape[0] > temp_df2.shape[0]:
                newdf_above50 = pd.concat([temp_df1,newdf_above50],ignore_index=True)

            else:
                newdf_above50 = pd.concat([temp_df2,newdf_above50],ignore_index=True)

        newdf_below50 = inputdata.loc[inputdata['subjectId'].isin(id_below50)]
        newdf = pd.concat([newdf_below50,newdf_above50],ignore_index=True)

        return newdf

    
    def histograms(self, data_description, plot_name, save_value = 0):
        """
        This method is used to plot histograms
        
        Function Parameters:
        data_description: the dataframe generated by the dataDescribe function 
        type: pandas DataFrame

        plot_name: name of the column whos data we wish to visualize

        save_value: flag to check whether to save the plot as an image or not

        Return:
        The output is histogram 
        """

        days = []
        for i in data_description['Timestamp days']:
            days.append(i.days);

        if plot_name == '# of days':
            fig = sns.distplot(days, kde = False, hist_kws=dict(edgecolor="k", linewidth=1), bins=10, color='palegreen')
            fig.set_xlabel('Number of Days', weight='bold', fontsize=16)
            fig.set_ylabel('Frequency', weight='bold', fontsize=16)
            sns.despine()
            if save_value == 1:
                plt.savefig(self.cwd+'/GlucoCheck/plots/Histogram - Number of days.png')

        if plot_name == '% of missing values':
            fig = sns.distplot(data_description['% of missing values'],kde = False, hist_kws=dict(edgecolor="k", linewidth=1), color='palegreen')
            fig.set_xlabel('Percent of missing values', weight='bold', fontsize=16)
            fig.set_ylabel('Frequency', weight='bold', fontsize=16)
            sns.despine()
            if save_value == 1:
                plt.savefig(self.cwd+'/GlucoCheck/plots/Histogram - % of missing values.png')

        if plot_name == 'Avg gap size':
            fig = sns.distplot(data_description['Avg gap size'],kde = False, hist_kws=dict(edgecolor="k", linewidth=1), color='palegreen')
            fig.set_xlabel('Percent of missing valuess', weight='bold', fontsize=16)
            fig.set_ylabel('Frequency', weight='bold', fontsize=16)
            sns.despine()
            if save_value ==1 :
                plt.savefig(self.cwd+'/GlucoCheck/plots/Histogram - Avg gap size.png')

        if plot_name == '# of missing values':
            fig = sns.distplot(data_description['# of missing values'], kde = False, hist_kws=dict(edgecolor="k", linewidth=1), bins=10, color='palegreen')
            fig.set_xlabel('# of Missing Values', weight='bold', fontsize=16)
            fig.set_ylabel('Frequency', weight='bold', fontsize=16)
            sns.despine()
            if save_value == 1:
                plt.savefig(self.cwd+'/GlucoCheck/plots/Histogram - Number of missing values.png')

        if plot_name == '# of readings':
            fig = sns.distplot(data_description['# of readings'], kde = False, hist_kws=dict(edgecolor="k", linewidth=1), bins=10, color='palegreen')
            fig.set_xlabel('Number of readings', weight='bold', fontsize=16)
            fig.set_ylabel('Frequency', weight='bold', fontsize=16)
            sns.despine()
            if save_value == 1:
                plt.savefig(self.cwd+'/GlucoCheck/plots/Histogram - Number of days.png')


    def barplots(self, data_description, plot_name, save_value = 0):
        """
        This method is used to plot barplots
        
        Function Parameters:
        data_description: the dataframe generated by the dataDescribe function 
        type: pandas DataFrame

        plot_name: name of the column whos data we wish to visualize

        save_value: flag to check whether to save the plot as an image or not

        Return:
        The output is barplots 
        """
        # sns.set_theme()
        days = []
        for i in data_description['Timestamp days']:
            days.append(i.days)

        # if plot_name == 'All':
        #     fig, axes = plt.subplots(3, 1, figsize=(20, 39))
        #     chart = sns.barplot(ax=axes[0], x = data_description['Subject ID'], y = days, color = 'skyblue', edgecolor="k", linewidth=0.5)
        #     chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize = 14);
        #     plt.rc('ytick', labelsize=15)
        #     fig.axes[0].set_xlabel('Subject ID', weight='bold', fontsize = 16)
        #     fig.axes[0].set_ylabel('Num of days', weight='bold', fontsize = 16)

        #     chart = sns.barplot(ax=axes[1], x = data_description['Subject ID'], y = data_description['% of missing values'], color = 'skyblue', edgecolor="k", linewidth=0.5)
        #     chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize = 14);
        #     fig.axes[1].set_xlabel('Subject ID', weight='bold', fontsize = 16)
        #     fig.axes[1].set_ylabel('Percent of missing values', weight='bold', fontsize = 16)

        #     chart = sns.barplot(ax=axes[2], x = data_description['Subject ID'], y = data_description['Avg gap size'], color = 'skyblue', edgecolor="k", linewidth=0.5)
        #     chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize = 14);
        #     fig.axes[2].set_xlabel('Subject ID', weight='bold', fontsize = 16)
        #     fig.axes[2].set_ylabel('Average gap size', weight='bold', fontsize = 16)
        #     sns.despine()

        if plot_name == '# of days':
            fig, axes = plt.subplots(1, 1, figsize=(20, 13))
            chart = sns.barplot(x = data_description['Subject ID'], y = days, color = 'palegreen', edgecolor="k", linewidth=0.5)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize = 14);
            plt.rc('ytick', labelsize=15)
            fig.axes[0].set_xlabel('Subject ID', weight='bold', fontsize = 16)
            fig.axes[0].set_ylabel('Num of days', weight='bold', fontsize = 16)
            sns.despine()
            if save_value == 1:
                fig.savefig(self.cwd+'/GlucoCheck/plots/Barplot - Number of missing values.png')

        if plot_name == '% of missing values':
            fig, axes = plt.subplots(1, 1, figsize=(20, 13))
            chart = sns.barplot(x = data_description['Subject ID'], y = data_description['% of missing values'], color = 'palegreen', edgecolor="k", linewidth=0.5)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize = 14);
            fig.axes[0].set_xlabel('Subject ID', weight='bold', fontsize = 16)
            fig.axes[0].set_ylabel('Percent of missing values', weight='bold', fontsize = 16)
            sns.despine()
            if save_value == 1:
                fig.savefig(self.cwd+'/GlucoCheck/plots/Barplot - Percent of missing values.png')

        if plot_name == 'Avg gap size':
            fig, axes = plt.subplots(1, 1, figsize=(20, 13))
            chart = sns.barplot(x = data_description['Subject ID'], y = data_description['Avg gap size'], color = 'palegreen', edgecolor="k", linewidth=0.5)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize = 14);
            fig.axes[0].set_xlabel('Subject ID', weight='bold', fontsize = 16)
            fig.axes[0].set_ylabel('Average gap size', weight='bold', fontsize = 16)
            sns.despine()
            if save_value == 1:
                fig.savefig(self.cwd+'/GlucoCheck/plots/Barplot - Avg gap size.png')


    def classifier(self, individual, training_data=all_data):
        """
        This method is used to classify an individual as diabetic, prediabetic or healthy
        
        Function Parameters:
        individual: the dataframe of the individual to be classified


        training_data: default dataframe we have type: pandas DataFrame

        Return:
        Class label - diabetic, prediabetic or healthy
        """
        meta = self.metadata
        meta = meta[meta.status != 'NAN']
        meta = meta[meta.status != 'NAN']
        training_data = training_data.loc[training_data['subjectId'].isin(meta.ID)]

        trainSubjects =  ['ID01','ID02','ID03','ID11','ID12','ID13','1636-69-001','1636-69-091','2133-004','2133-018',
                 'ID21', 'ID22','ID23','1636-69-026', '1636-69-032','1636-69-090','1636-69-114','1636-70-1005','2133-015','2133-017',
                 'ID29','ID30','ID31','1636-69-028','1636-69-035','1636-69-048','1636-69-053','1636-69-060','1636-69-064','1636-69-100',
                 '2133-039','1636-70-1010','1636-69-069'
                ]

        testSubjects = ['2133-039','1636-70-1010','1636-69-064', '1636-69-001','2133-004',
                        '1636-69-026','1636-69-032','1636-69-028','1636-69-035','ID01','ID02',
                         'ID22','ID23','ID30','ID31']

        tsdata = []
        for subjectId, d in training_data.groupby('subjectId'):
            l = []
            l.append(subjectId)

            for gv in d.GlucoseValue:
                l.append(gv)
            tsdata.append(l[:580])

        tsdf = pd.DataFrame.from_records(tsdata)

        # tsdf['Y'] = meta['status'].tolist()

        tsdf = pd.merge(tsdf,meta,left_on=0,right_on='ID')
        del tsdf['ID']

        # tsdf = tsdf.loc[tsdf[0].isin(trainSubjects)]
        tsdf = tsdf.set_index([0],drop=True)

        trainingSet = tsdf.loc[tsdf.index.isin(trainSubjects)]
        trainingSet = trainingSet.reset_index(drop=True)

        testingSet = tsdf.loc[tsdf.index.isin(testSubjects)]
        testingSet = testingSet.reset_index(drop=True)

        X_train = trainingSet.drop(['status'], axis=1)
        y_train = trainingSet['status']

        l=[]
        test = []

        for gv in individual.GlucoseValue:
            l.append(gv)
        test.append(l[:579])

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train,y_train)

        X_test = testingSet.drop(['status'], axis=1)
        y_test = testingSet['status']

        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=["diabetic","pre-diabetic","non-diabetic"])
        print("Confusion Matrix:\n")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["diabetic","pre-diabetic","non-diabetic"])
        disp = disp.plot()
        plt.show()

        y_pred = clf.predict(test)

        return y_pred



#==================================================================================================================
#   GVI's
#==================================================================================================================

    def gvIndices(self, data):
        """
            This method provides a consolidated  table of all the glucose variability indices of all the individuals in the default data (The default data set for the training includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package, and the Ohio University dataset)

            Function Parameters:
            data:The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package, and the Ohio University with the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            type: pandas DataFrame

            Return:
            A consolidated table of all the glucose variability indices of all the individuals. This table is also saved as a CSV file.
        """
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

            LBGI, HBGI, BGRI = self.bgri(df, units = 'mg');

            GRADE , HypoG_P, EuG_P, HyperG_P = self.grade(df, units='mg');

            j_index = self.j_index(df, units="mg");

            Mvalue = self.m_value(df, 'mg', 120);

            MAG = self.mag(df);

            GVP = self.gvp(df, units='mg');

            GMI = self.gmi(df, units='mg');

            LAGE, MAX, MIN = self.lage(df);

            HBA1C = self.ehba1c(df);

            m, sd, cv, iqr = self.sumstats(df);

            sdrc = self.rc(df);

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

            dt = self.dt(df);

            TAR_VH, TAR_H, TIR, TBR_L, TBR_VL = self.tir(df, units='mg');

            xx = self.subSample(df);
            hypo, hyper = self.variabilityEpisodes(xx, "mg");

            igc, hypoglycemicIndex, hyperglycemicIndex = self.IGC(df, 'mg');

            li = self.glucoseLiabilityIndex(df, 'mg');

            # adrr_val = self.adrr(df, 'mg')
            adrr_daily = []
            for Date, xx in df.groupby('Date'):
                xx = xx.reset_index(drop=True)
                z = self.adrr(xx,'mg')
                adrr_daily.append(z)

            adrr_val = round(mean(adrr_daily),2)

            modd_val = self.modd(df);

            conga_1 = self.congaN(df, 1);
            conga_2 = self.congaN(df, 2);
            conga_4 = self.congaN(df, 4);

            mage, mage_daily = self.mageCalculations(df, 1)


            temp_df = pd.DataFrame({'Subject ID':[subjectId], "ADDR": [adrr_val], 'BGRI':[round(BGRI,3)], 'LBGI':[round(LBGI,3)], 'HBGI':[round(HBGI,3)], "CONGA1": [conga_1], "CONGA2": [conga_2], "CONGA4": [conga_4], 'DT':[round(dt,3)], 'HBA1C':[round(HBA1C,3)], 'GFI':[round(gfi,3)], 'GCF':[round(gcf,3)], "Liability Index": [li], 'GMI':[round(GMI,3)],  'GRADE':[round(GRADE,3)], 'HypoG_P':[round(HypoG_P,3)],'EuG_P':[round(EuG_P,3)], 'HyperG_P':[round(HyperG_P,3)], 'GVP':[round(GVP,3)], "IGC": [igc], "Hypoglycemic Index": [hypoglycemicIndex], "Hyperglycemic Index": [hyperglycemicIndex], 'J Index':[round(j_index,3)], 'LAGE':[round(LAGE,3)], 'Mvalue':[round(Mvalue,3)], 'MAG':[round(MAG,3)], "MODD": [modd_val], 'PGS':[round(pgs_value,3)], 'SDRC':[round(sdrc,3)], 'MEAN':[round(m,3)], 'STD-DEV':[round(sd,3)],'CV':str([round(cv,3)])+"%", 'IQR':[round(iqr,3)], 'MAX':[round(MAX,3)], 'MIN':[round(MIN,3)], 'TAR_VH(%)': [round(TAR_VH,3)], 'TAR_H(%)': [round(TAR_H,3)], 'TIR(%)': [round(TIR,3)], 'TBR_L(%)': [round(TBR_L,3)], 'TBR_VL(%)': [round(TBR_VL,3)], 'Hypoglycemic Episodes': [hypo], 'Hyperglycemic Episodes': [hyper], 'MAGE': [round(mage,3)], 'MAGE Daily': [round(mage_daily,3)]})
            data_description = pd.concat([data_description,temp_df],ignore_index=True)

        # data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)

        return (data_description)

        data_description.to_csv(self.cwd+"/GlucoCheck/Data/Glucose Indices.csv")


    def individualGvIndices(self, uid):
        """
        This method provides a consolidated  table of all the glucose variability indices of all the individuals in the default data (The default data set for the training includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package, and the Ohio University dataset)

        Function Parameters:
        uid: The subject ID of the user to plot  type: String

        data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package, and the Ohio University with the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        This data is split based on the subject ID
        type: pandas DataFrame

        Return:
        A consolidated table of all the glucose variability indices of the individuals.
        """
        df = data[data['subjectId']==str(uid)]
        df = self.fullDaysOnly(df)
        df['Display Time'] = pd.to_datetime(df['Display Time'])

        df=df.reset_index(drop=True)

        dates = []
        for i in range(len(df.index)):
            dates.append(df['Display Time'][i].date())
        df['Date'] = dates

        gfi, gcf = self.gfi(df);

        LBGI, HBGI, BGRI = self.bgri(df, units = 'mg');

        GRADE , HypoG_P, EuG_P, HyperG_P = self.grade(df, units='mg');

        j_index = self.j_index(df, units="mg");

        Mvalue = self.m_value(df, 'mg', 120);

        MAG = self.mag(df);

        GVP = self.gvp(df, units='mg');

        GMI = self.gmi(df, units='mg');

        LAGE, MAX, MIN = self.lage(df);

        HBA1C = self.ehba1c(df);

        m, sd, cv, iqr = self.sumstats(df);

        sdrc = self.rc(df);

        start = df.Date[0]
        end = start+timedelta(days=7)
        last_date = df.Date.iloc[-1]
        pgs_weekly = []
        li_weekly = []
        while end <= last_date:
            xy = pd.DataFrame()
            mask = (df['Date'] >= start) & (df['Date'] <= end)
            xy = pd.concat([xy, df.loc[mask]],ignore_index=True)
            pgs_weekly.append(self.pgs(xy, units='mg'))
            li_weekly.append(self.li(xy, units='mg'))
            start = end+timedelta(days=1)
            end = start+timedelta(days=7)

        xy = pd.DataFrame()
        end = last_date
        mask = (df['Date'] >= start) & (df['Date'] <= end)
        xy = pd.concat([xy, df.loc[mask]],ignore_index=True)
        pgs_weekly.append(self.pgs(xy, units='mg'))
        li_weekly.append(self.li(xy, units='mg'))

        pgs_value = mean(pgs_weekly)
        li = mean(li_weekly)

        dt = self.dt(df);

        TAR_VH, TAR_H, TIR, TBR_L, TBR_VL = self.tir(df, units='mg');

        xx = self.subSample(df);
        hypo, hyper = self.variabilityEpisodes(xx, "mg");

        igc, hypoglycemicIndex, hyperglycemicIndex = self.IGC(df, 'mg');

        adrr_daily = []
        for Date, xx in df.groupby('Date'):
            xx = xx.reset_index(drop=True)
            z = self.adrr(xx,'mg');
            adrr_daily.append(z)

        adrr_val = round(mean(adrr_daily),2)

        modd_val = self.modd(df);

        conga_1 = self.congaN(df, 1);
        conga_2 = self.congaN(df, 2);
        conga_4 = self.congaN(df, 4);

        mage, mage_daily = self.mageCalculations(df, 1)

        data_description = pd.DataFrame({'Subject ID':[uid], "ADDR": [adrr_val], 'BGRI':[round(BGRI,3)], 'LBGI':[round(LBGI,3)], 'HBGI':[round(HBGI,3)], "CONGA1": [conga_1], "CONGA2": [conga_2], "CONGA4": [conga_4], 'DT':[round(dt,3)], 'HBA1C':[round(HBA1C,3)], 'GFI':[round(gfi,3)], 'GCF':[round(gcf,3)], "Liability Index": [li], 'GMI':[round(GMI,3)],  'GRADE':[round(GRADE,3)], 'HypoG_P':[round(HypoG_P,3)],'EuG_P':[round(EuG_P,3)], 'HyperG_P':[round(HyperG_P,3)], 'GVP':[round(GVP,3)], "IGC": [igc], "Hypoglycemic Index": [hypoglycemicIndex], "Hyperglycemic Index": [hyperglycemicIndex], 'J Index':[round(j_index,3)], 'LAGE':[round(LAGE,3)], 'Mvalue':[round(Mvalue,3)], 'MAG':[round(MAG,3)], "MODD": [modd_val], 'PGS':[round(pgs_value,3)], 'SDRC':[round(sdrc,3)], 'MEAN':[round(m,3)], 'STD-DEV':[round(sd,3)],'CV':str([round(cv,3)])+"%", 'IQR':[round(iqr,3)], 'MAX':[round(MAX,3)], 'MIN':[round(MIN,3)], 'TAR_VH(%)': [round(TAR_VH,3)], 'TAR_H(%)': [round(TAR_H,3)], 'TIR(%)': [round(TIR,3)], 'TBR_L(%)': [round(TBR_L,3)], 'TBR_VL(%)': [round(TBR_VL,3)], 'Hypoglycemic Episodes': [hypo], 'Hyperglycemic Episodes': [hyper], 'MAGE': [round(mage,3)], 'MAGE Daily': [round(mage_daily,3)]})
        # data_description = data_description.iloc[::-1]

        data_description = data_description.set_index(['Subject ID'], drop=True)

        return (data_description)


    def gfi(self, x):
        """
            Glucose Fluctuation Index and Glucose Coefficient of Fluctuation
            The GFI is based on consecutive glucose differences, where consecutive differences in GFI are squared prior to finding their mean and taking the square root. The potential benefit is that differences are weighted individually, giving more importance to the greatest ones, which are likely to be more detrimental. GCF is computed as the ratio of GFI to the mean of input glucose values.

            DESCRIPTION:
            The function takes in a sequence of continuous glucose values,  and computes glucose fluctuation index (GFI)  and the glucose coefficient of fluctuation (GCF). This function accepts data given either in mmol/L or mg/dl.

            FUNCTION PARAMETERS:
            x: Pandas data frame, in the first column, is given Pandas time stamp, in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame

            RETURN:
            The output is numeric values for GFI and GCF accordingly;

            REFERENCES:
            - Le Floch J, Kessler L (2016). “Glucose variability: comparison of different indices during continuous glucose monitoring in diabetic patients.” Journal of diabetes science and technology, 10(4), 885–891.
        """
        N = len(x)
        S = 0
        for i in range(0,N-1):
            S = S + (x.iloc[i, 2]  - x.iloc[(i+1), 2]) ** 2

        gfi = np.sqrt(S/N)
        gcf = gfi/np.mean(x.iloc[:,2])
        # return pd.DataFrame({'GFI':[gfi], 'GCF':[gcf]})
        if math.isinf(gfi):
            # print("Error calculating GFI for: "+str(x["subjectId"]))
            gfi = 0.0
        elif math.isinf(gcf):
            # print("Error calculating GCF for: "+str(x["subjectId"]))
            gcf = 0.0

        return round(gfi,2), round(gcf,2)


    def bgri(self, x, units):
        """
            Blood Glucose Risk Index
            LBGI is a measure of the frequency and extent of low blood glucose (BG) readings; HBGI is a measure of the frequency and extent of high BG readings; BGRI is a measure for the overall risk of extreme BG equal to LBGI + HBGI.

            The LBGI has been validated as a predictor of severe hypoglycemia, while the HBGI has been related to risk for hyperglycemia and HbA1c; Both indices demonstrate high sensitivity to changes in glycemic profiles and metabolic control, as well as high sensitivity to the effects of treatment. Larger values of LBGI and HBGI indicate a higher risk for hypoglycemia and hyperglycemia, respectively.
            Although originally derived using self-monitored blood glucose data, these parameters have been adapted to continuous interstitial glucose monitoring data. Correlations between LBGI and subsequent hypoglycemia and between HBGI and HbA1c have been reported.

            The LBGI and the HBGI are non-negative numbers; each index and their sum could range theoretically between 0 and 100.

            DESCRIPTION:
            Takes in a sequence of continuous glucose values and computes: low blood glucose index (LBGI), high blood glucose index (HBGI), and overall blood glucose risk index (BGRI). This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS:
            x: Pandas data frame, in the first column, is given Pandas time stamp, in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame
            unit: should be set either to "mmol" or to "mg"   type: string


            RETURN:
            The output is numeric values for LBGI, HBGI and BGRI accordingly; details LBGI is a measure of the frequency and extent of low blood glucose (BG) readings;

            REFERENCES:
            - Service FJ (2013). “Glucose variability.” Diabetes, 62(5), 1398.
            - Kovatchev BP, Clarke WL, Breton M, Brayman K, McCall A (2005). “Quantifying temporal glucose variability in diabetes via continuous glucose monitoring: mathematical methods and clinical application.” Diabetes technology & therapeutics, 7(6), 849–862.
        """
        if (units == 'mg'):
            fBG = 1.509*((np.log(   x.GlucoseValue) )**1.084  - 5.381)
        elif (units=='mmol'):
            fBG = 1.509*((np.log(18*x.GlucoseValue) )**1.084  - 5.381)
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
            # print("Error calculating LBGI for: "+str(x["subjectId"]))
            LBGI = 0.0
        elif math.isinf(HBGI):
            # print("Error calculating HBGI for: "+str(x["subjectId"]))
            HBGI = 0.0
        elif math.isinf(BGRI):
            # print("Error calculating BGRI for: "+str(x["subjectId"]))
            BGRI = 0.0

        return round(LBGI,2), round(HBGI,2), round(BGRI,2)


    def grade(self, x, units):
        """
            Glycaemic Risk Assessment Diabetes Equation
            GRADE is a score derived to summarize the degree of risk associated with a certain glucose profile. Qualitative risk scoring for a wide range of glucose levels inclusive of marked hypoglycemia and hyperglycemia is obtained based on a committee of diabetes practitioners. The calculated score can range from 0 -- meaning no risk to 50 -- meaning maximal risk. The structure of the formula is designed to give a continuous curvilinear approximation with a nadir at 4.96 mmol/L (90 mg/dL) and high adverse weighting for both hyper- and hypoglycemia. The contribution of hypoglycemia, euglycemia, and hyperglycemia to the GRADE score are expressed as percentages: e.g. GRADE (hypoglycemia %, euglycemia %, hyperglycemia %), which are defined as:
            <3.9 mmol/L (70 mg/dL) hypoglycaemia;
            3.9 - 7.8mmol/L (70–140 mg/dL) euglycemia;
            and >7.8 mml/L (140 mg/dL) hyperglycemia.


            DESCRIPTION:
            Takes in a sequence of continuous glucose values and computes Glycaemic Risk Assessment Diabetes Equation (GRADE) score. This function accepts data given either in mmol/L or mg/dL.

            FUNCTION PARAMETERS:
            x: Pandas data frame, in the first column, is given Pandas time stamp, in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame
            units:should be set either to "mmol" or to "mg"   type: string


            RETURN:
            The output is numeric values for GRADE and percentages expressing risk calculated from hypoglycemia, euglycemia, and hyperglycemia;

            REFERENCES:
            - Service FJ (2013). “Glucose variability.” Diabetes, 62(5), 1398.
            - Hill N, Hindmarsh P, Stevens R, Stratton I, Levy J, Matthews D (2007). “A method for assessing quality of control from glucose profiles.” Diabetic medicine, 24(7), 753–758.

        """
        if (units == 'mg'):
            a = 1/18
            g = np.append(np.where(x.iloc[:, 2] <= 37)[0], np.where(x.iloc[:, 2] >= 630)[0])
            hypo = np.where(x.iloc[:, 2] < 70)[0]
            eu = np.where((x.iloc[:, 2] >= 70) & (x.iloc[:, 2]<=140))[0]
            hyper = np.where(x.iloc[:, 2] > 140)[0]
        elif (units=='mmol'):
            a = 1
            g = np.append(np.where(x.iloc[:, 2] <= 2.06)[0], np.where(x.iloc[:, 2] >= 33.42)[0])
            hypo = np.where(x.iloc[:, 2]<3.9)[0]
            eu = np.where(x.iloc[:, 2]>=3.9 & x.iloc[:, 2] <=7.8)[0]
            hyper = np.where(x.iloc[:, 2]>7.8)[0]
        else:
            print('units should be either mmol or mg')
            return 0

        grd = 425*( np.log10( np.log10(a*x.iloc[:, 2]) ) + 0.16) ** 2


        if (len(g)>0):  # GRADE is designed to operate for BG ranges between 2.06 (37 mg/dl) and 33.42 mmol/l (630 mg/dl).
            grd[g] = 50 # Values outside this range are ascribed a GRADE value of 50.

        tmp = (np.mean(grd), len(hypo)/len(x)*100, len(eu)/len(x)*100, len(hyper)/len(x)*100)

        GRADE = np.mean(grd)
        HypoG_P = sum(hypo)/sum(grd)*100
        EuG_P = sum(eu)/sum(grd)*100
        HyperG_P = sum(hyper)/sum(grd)*100

        # return pd.DataFrame({'GRADE':[np.mean(grd)], 'HypoG%':[len(hypo)/len(x)*100], 'EuG%':[len(eu)/len(x)*100], 'HyperG%':[len(hyper)/len(x)*100]})
        if math.isinf(GRADE):
            # print("Error calculating GRADE for: "+str(x["subjectId"]))
            GRADE = 0.0
        elif math.isinf(HypoG_P):
            # print("Error calculating HypoG_P for: "+str(x["subjectId"]))
            HypoG_P = 0.0
        elif math.isinf(EuG_P):
            # print("Error calculating EuG_P for: "+str(x["subjectId"]))
            EuG_P = 0.0
        elif math.isinf(HyperG_P):
            # print("Error calculating HyperG_P for: "+str(x["subjectId"]))
            HyperG_P = 0.0

        return round(GRADE,2) , round(HypoG_P,2), round(EuG_P,2), round(HyperG_P,2)


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

        j = a*(np.mean(x.iloc[:, 2]) + np.std(x.iloc[:, 2])) ** 2

        # return pd.DataFrame({'J-index':[j]})
        if math.isinf(j):
            # print("Error calculating J-Index for: "+str(x["subjectId"]))
            j = 0.0
        return round(j,2)


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
            PG = x.GlucoseValue
        elif (units=='mmol'):
            PG = 18*x.GlucoseValue
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
            # print("Error calculating Mvalue for: "+str(x["subjectId"]))
            Mvalue = 0.0

        return round(Mvalue,2)


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
        S = np.abs(np.sum(x.iloc[:, 2].diff()))
        n = len(x)-1
        total_T = (x.iloc[n,1] - x.iloc[0, 1])/np.timedelta64(1,'h')
        MAG = S/total_T
        # return pd.DataFrame({'MAG':[MAG]})

        if math.isinf(MAG):
            # print("Error calculating MAG for: "+str(x["subjectId"]))
            MAG = 0.0

        return round(MAG,2)


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

        dt = x.iloc[:, 1].diff()/np.timedelta64(1,'m') # assuming that sampling can not necessarily be equally spaced
        dy = x.iloc[:, 2].diff()

        L = np.sum(np.sqrt(dt**2 + dy**2))
        L_0 = np.sum(dt)

        GVP = (L/L_0 -1) *100
        # return pd.DataFrame({'GVP(%)':[GVP]})

        if math.isinf(GVP):
            # print("Error calculating GVP for: "+str(x["subjectId"]))
            GVP = 0.0

        return round(GVP,2)


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
            GMI = 3.31 + 0.02392 * np.mean(x.iloc[:, 2])
            # return pd.DataFrame({'GMI(%)': [GMI]})
            return GMI
        elif (units=='mmol'):
            GMI = 12.71 + 4.70587 * np.mean(x.iloc[:, 2])
            # return pd.DataFrame({'GMI(%)': [GMI]})
            return round(GMI,2)
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
        MIN = np.min(x.iloc[:, 2])
        MAX = np.max(x.iloc[:, 2])
        LAGE = MAX - MIN
        # return pd.DataFrame({'LAGE': [LAGE], 'MAX': [MAX], 'MIN':[MIN]})
        return round(LAGE,2), round(MAX,2), round(MIN,2)


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
        HBA1C = (np.mean(x.iloc[:, 2]) + 46.7)/28.7
        # return pd.DataFrame({'eHbA1c': [HBA1C]})

        if math.isinf(HBA1C):
            # print("Error calculating HBA1C for: "+str(x["subjectId"]))
            HBA1C = 0.0

        return round(HBA1C,2)


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
        m = np.mean(x.iloc[:, 2])
        sd = np.std(x.iloc[:, 2])
        cv = (sd/m)
        q75, q25 = np.percentile(x.iloc[:, 2], [75 ,25])
        iqr = q75 - q25

        # return pd.DataFrame({'Mean': [m], 'SD':[sd], 'CV': [cv], 'IQR': [iqr]})
        return round(m,2), round(sd,2), 100*(round(cv,2)), round(iqr,2)


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
        dt = x.iloc[:, 1].diff()/np.timedelta64(1,'m')
        dy = x.iloc[:, 2].diff()

        sdrc = np.std(dy/dt)
        # return pd.DataFrame({'SD of RC': [sdrc]})

        if math.isinf(sdrc):
            # print("Error calculating SDRC for: "+str(x["subjectId"]))
            sdrc = 0.0

        return round(sdrc,2)


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

        N54 = len(x[x.iloc[:,2]<=54])
        F_54H = 0.5 + 4.5 * (1 - np.exp(-0.81093*N54))

        N70 = len(x[x.iloc[:,2]<70]) - N54

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
        TIR  =  len(x) - len(x[x.iloc[:,2]<70].iloc[:,2]) - len(x[x.iloc[:,2]>180].iloc[:,2])
        PTIR = TIR*100/lx

        F_PTIR = 1 + 9/(1 + np.exp(0.0833*(PTIR - 55.04)))

        MG = np.mean(x.iloc[:, 2])
        F_MG = 1 + 9 * ( 1/(1 + np.exp(0.1139*(MG-72.08))) + 1/(1 + np.exp(-0.09195*(MG-157.57))) )

        PGS = F_GVP + F_MG + F_PTIR + F_H
        # PGS.columns=['PGS']

        if math.isinf(PGS):
            # print("Error calculating PGS for: "+str(x["subjectId"]))
            PGS = 0.0

        return round(PGS,2)


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
        dy = np.sum(np.abs(x.iloc[:, 2].diff()))
        return round(dy,2)
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
            TAR_VH = len(x[x.iloc[:,2]> 250])/N*100
            TAR_H  = len(x[(x.iloc[:,2]>= 181) & (x.iloc[:,2]<= 250)])/N*100
            TIR    = len(x[(x.iloc[:,2]>= 70) & (x.iloc[:,2]<= 180)])/N*100
            TBR_L  = len(x[(x.iloc[:,2]>= 54) & (x.iloc[:,2]<= 69)])/N*100
            TBR_VL = len(x[x.iloc[:,2]< 54])/N*100
            # return pd.DataFrame({'TAR_VH(%)': [TAR_VH], 'TAR_H(%)': [TAR_H], 'TIR(%)': [TIR], 'TBR_L(%)': [TBR_L], 'TBR_VL(%)': [TBR_VL]})
            return TAR_VH, TAR_H, TIR, TBR_L, TBR_VL
        elif (units=='mmol'):
            N = len(x)
            TAR_VH = len(x[x.iloc[:,2]> 13.9])/N*100
            TAR_H  = len(x[(x.iloc[:,2]>= 10.1) & (x.iloc[:,2]<= 13.9)])/N*100
            TIR    = len(x[(x.iloc[:,2]>= 3.9) & (x.iloc[:,2]<= 10.0)])/N*100
            TBR_L  = len(x[(x.iloc[:,2]>= 3.0) & (x.iloc[:,2]<= 3.8)])/N*100
            TBR_VL = len(x[x.iloc[:,2]< 3.0])/N*100
            # return pd.DataFrame({'TAR_VH(%)': [TAR_VH], 'TAR_H(%)': [TAR_H], 'TIR(%)': [TIR], 'TBR_L(%)': [TBR_L], 'TBR_VL(%)': [TBR_VL]}
            return round(TAR_VH,2), round(TAR_H,2), round(TIR,2), round(TBR_L,2), round(TBR_VL,2)
        else:
            return print('units should be either mmol or mg')


    def variabilityEpisodes(self, df, unit):
        """
        Number of hypoglycemic (glucose concentrations of less than 3.0 mmol/L (54 mg/dL))and number of hyperglycemic (glucose concentration of more than 13.9 mmol/L (250 mg/dL)) episodes that last at least 15min. Hypoglycemic Excursions and Hyperglycemic Excursions are computed for each day and then averaged across all days within individual time series.

        DESCRIPTION:
        Takes in a sequence of continuous glucose values and computes Hypoglycemic Excursions and Hyperglycemic Excursions. This function works with data given either in mmol/L or mg/dL.

        FUNCTION PARAMETERS:
        df: Pandas data frame, in the first column, is given Pandas time stamp, in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame
        units   should be set either to "mmol" or to "mg"   type: string

        RETURN:
        The output a numeric value for average Hypoglycemic Excursions and Hyperglycemic Excursions.

        REFERENCES:
        - T. Battelino, T. Danne, R. M. Bergenstal, S. A. Amiel, R. Beck, T. Biester,E. Bosi, B. A. Buckingham, W. T. Cefalu, K. L. Close, et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range.Diabetes Care, 42(8):1593–1603, 2019.
        """
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

        return round(hypoglycemic_episodes,2), round(hyperglycemia_episodes,2)


    def IGC(self, df, unit, lltr = 80, ultr = 140, a = 1.1, b = 2.0, c = 30, d = 30):
        """
        Index of Glycemic Control
        Sum of Hyperglycemia Index and Hypoglycemia Index

        DESCRIPTION:
        Takes in a sequence of continuous glucose values and computes the IGC, Hypoglycemic Excursions and Hyperglycemic Excursions. This function works with data given either in mmol/L or mg/dL.

        FUNCTION PARAMETERS:

        df: Pandas data frame, in the first column, is given Pandas time stamp,in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame
        units: should be set either to "mmol" or to "mg"   type: string
        lltr: constant. default is 80.    type: int
        ultr: constant. default is 140.   type: int
        a: constant. default is 1.1.   type: int
        b: constant. default is 2.     type: int
        c: constant. default is 30.    type: int
        d: constant. default is 30.    type: int

        RETURN:
        The output a numeric value for IGC, Hypoglycemic index, Hyperglycemic index

        REFERENCES:
        -“Interpretation of Continuous Glucose Monitoring Data: Glycemic Variability and Quality of Glycemic Control” by David Rodbard, M.D. (see appendix 2). This function must output three measures: IGC, Hypoglycemic index, Hyperglycemic index.
        """
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
        return round(igc,2), round(hypoglycemicIndex,2), round(hyperglycemicIndex,2)


    def glucoseLiabilityIndex(self,data, unit):
        """
        DESCRIPTION:
        Takes in a sequence of continuous glucose values and computes the Liability Index. This function works with data given either in mmol/L or mg/dL.

        FUNCTION PARAMETERS:
        data:Pandas data frame, in the first column, is given Pandas time stamp,in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame

        units   should be set either to "mmol" or to "mg"   type: string

        RETURN:
        The output a numeric value for Liability index

        REFERENCES:
        -“E. A. Ryan, T. Shandro, K. Green, B. W. Paty, P. A. Senior, D. Bigam, A. J. Shapiro, and M.-C. Vantyghem. Assessment of the severity of hypoglycemia and glycemic lability in type 1 diabetic subjects undergoing islet transplantation. Diabetes, 53(4):955–962, 2004”: https://diabetes.diabetesjournals.org/content/53/4/955.long


        """
        data = self.hourlySamples(data)
        if unit == 'mg':
            data['GlucoseValue'] = data['GlucoseValue']/18
        gli = np.sum(np.power(data['GlucoseValue'][i] - data['GlucoseValue'][i+1],2) for i in range(0, len(data.index)-1))
        return round(gli,2)


    def adrr(self, x, units):
        """
        Average Daily Risk Range
        The average sum of |HBGI for maximum glucose| plus |LBGI for minimum glucose| for each day.
        High Blood Glucose Index (HBGI), Low Blood Glucose Index (LBGI)

        DESCRIPTION:
        Takes in a sequence of continuous glucose values and computes the ADRR. This function works with data given either in mmol/L or mg/dL.

        FUNCTION PARAMETERS:
        xx: Pandas data frame, in the first column, is given Pandas time stamp,in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame
        units: should be set either to "mmol" or to "mg"   type: string
        RETURN:
        The output a numeric value for ADRR

        REFERENCES:
        -“B. P. Kovatchev, E. Otto, D. Cox, L. Gonder-Frederick, and W. Clarke. Evaluation of a new measure of blood glucose variability in diabetes. Diabetes care, 29(11):2433–2438, 2006”: https://care.diabetesjournals.org/content/29/11/2433.long
        """
        if (units == 'mg'):
            fBG = 1.509*((np.log(x['GlucoseValue']))**1.084  - 5.381)
        elif (units=='mmol'):
            fBG = 1.509*((np.log(18*x['GlucoseValue']) )**1.084  - 5.381)
        else:
            return print('units should be either mmol or mg')
            return 0

        rBG = 10 * fBG ** 2 # called BG risk function
        s = np.sign(fBG)
        s_left = np.abs(s.where(s == -1, 0))
        rlBG = rBG * s_left # called BG risk function left branch

        s_right = s.where(s == 1, 0)
        rhBG = rBG * s_right # called BG risk function right branch

        ADRR = max(rlBG) + max(rhBG) # !!amend the code to output average across days !!!!!

        return ADRR

    
    def modd(self, data):
        """
        Mean of Daily Differences
        Mean difference between glucose values obtained at the same time of day on two consecutive days under standardized conditions

        DESCRIPTION:
        Takes in a sequence of continuous glucose values and computes the MODD. This function works with data given either in mmol/L or mg/dL.

        FUNCTION PARAMETERS:
        data: Pandas data frame, in the first column, is given Pandas time stamp, in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame

        RETURN:
        The output a numeric value for MODD

        REFERENCES:
        -“C. McDonnell, S. Donath, S. Vidmar, G. Werther, and F. Cameron. A novel approach to continuous glucose analysis utilizing glycemic variation. Diabetes technology & therapeutics, 7(2):253–263, 2005”.
        """
        data['Display Time'] = data['Display Time'].dt.round('5min')

        times = []
        for i in range(len(data.index)):
            times.append(data['Display Time'][i].time())
        data['Time'] = times


        Modd = []
        s = []
        gvDiff = 0
        for Time, df in data.groupby('Time'):
            gvDiff = abs(df['GlucoseValue'] - df['GlucoseValue'].shift(-1))
            gvDiff = gvDiff.dropna()
            s.append((gvDiff))

        return np.round(np.mean(s),2)


    def congaN(self, df, n):
        """
        Continuous Overlapping Net Glycemic Action
        A measure of within-day glucose variability: SD of differences between any glucose value and another one exactly N hours later.

        DESCRIPTION:
        Takes in a sequence of continuous glucose values and computes the CONGA. This function works with a value 'n', that is the hour parameter that is usually 1, 2, or 4.

        FUNCTION PARAMETERS:
        df  Pandas data frame, in the first column, is given Pandas time stamp,
        in the second - numeric values of continuous glucose readings, and in the third - subject ID    type: pandas DataFrame
        n   hour parameter; usually 1, 2, or 4. type: int

        RETURN:
        The output a numeric value for CONGA

        REFERENCES:
        “C. McDonnell, S. Donath, S. Vidmar, G. Werther, and F. Cameron. A novel approach to continuous glucose analysis utilizing glycemic variation. Diabetes technology & therapeutics, 7(2):253–263, 2005”
        """
        if (not(n==1 or n==2 or n==4)):
            print("WARNING: Standard range for CONGA n values are 1,2 or 4 hours. CONGA measure might be  unreliable if observations are more than 4h apart.")
        day = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
        day = day.round("d")
        day = day.days

        df = df.set_index(['Display Time'])
        t = str(n*3600)+'s'
        gv = df['GlucoseValue'].resample(t).first()

        k = len(gv)

        frame = pd.DataFrame()
        frame['GV'] = gv
        frame['Dt'] = frame['GV'] - frame['GV'].shift(+1)
        frame = frame.fillna(0)

        dBar = sum(frame['Dt']) / k

        s = 0
        for i in frame['Dt']:
            s += (i-dBar)**2

        conga = math.sqrt(s/(k-1))

        return round(conga/day, 2)



    def mad_index(self, x):
        MAD = np.abs(np.sum(x['GlucoseValue'].diff())/len(x))
        # return pd.DataFrame({'MAD':[MAD]})
        return round(MAD,2)
        # Mean Absolute Difference
        # MAD was proposed as measures of glycemic variability and derived
        # from self-monitored consecutive blood glucose values over 24 h
        #
        # DESCRIPTION: Takes in a sequence of continuous glucose values
        # and computes mean absolute difference (MAD) of consecutive blood glucose values.
        # This function accepts data given either in mmol/L or mg/dL.
        #
        # FUNCTION PARAMETERS: x - is Pandas dataframe, in the first column is given subject ID,
        # in the second - Pandas timestamp, and in the third - numeric values of
        # continuous glucose readings taken e.g. over one day (24h);
        #
        # RETURN: Output is Pandas dataframe that contains numeric value for MAD.
        #
        # REFERENCES:
        # - Moberg E, Kollind M, Lins P, Adamson U (1993). “Estimation of blood-glucose variability
        # in patients with insulin-dependent diabetes mellitus.” Scandinavian journal of clinical
        # and laboratory investigation, 53(5), 507–514.


    def mageCalculations(self, df, stdev):
        """
        """
    
        #smoothening the data
        df = self.fullDay(df)
        df = self.smoothing(df)
        
        length = df['Display Time'].iloc[-1]-df['Display Time'].iloc[0]
        length = length.round("d")
        days = length.days
        
        df.GlucoseValue = df.GlucoseValue.round()
        
        glucs = df['GlucoseValue'].tolist()
        indices = [1*i for i in range(len(glucs))]
        
        x = indices
        gvs = glucs

        # detection of local minima and maxima

        # local min & max
        a = np.diff(np.sign(np.diff(gvs))).nonzero()[0] + 1      
        # local min
        valleys = (np.diff(np.sign(np.diff(gvs))) > 0).nonzero()[0] + 1 
        # local max
        peaks = (np.diff(np.sign(np.diff(gvs))) < 0).nonzero()[0] + 1 
        
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
        
        #stdev = 1
        turning_points = pd.DataFrame(columns=['Index', 'Timestamp', 'GlucoseValue', 'Type'])
        k=0
        for i in range(stdev,len(excursion_points.Index)-stdev):
            positions = [i-stdev,i,i+stdev]
            for j in range(0,len(positions)-1):
                if(excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]):
                    if(excursion_points.Type[positions[j]]=='P'):
                        if excursion_points.GlucoseValue[positions[j]]>=(excursion_points.GlucoseValue[positions[j+1]]+2):
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
                    else:
                        if (excursion_points.GlucoseValue[positions[j]]-2)<=excursion_points.GlucoseValue[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j]]
                            k+=1
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
      
        if len(turning_points.index)<10:
            turning_points = excursion_points.copy()

        turning_points = turning_points.drop_duplicates(subset= "Index", keep= "first")
        turning_points = turning_points.reset_index(drop=True)
        turning_points_count = len(turning_points.index)
        
        
        mage = turning_points.GlucoseValue.sum()/turning_points_count
        #print(mage)
        mage_per_day = mage/days
        #print(mage_per_day)
        
        return mage, mage_per_day

    
#==================================================================================================================
#   Helper Methods
#==================================================================================================================
    
    def meanCalculations(self, xx):
        """
        This function calculates the daytime and nighttime mean glucose values

        Function Parameters:
        data: The data of an individual's glucose readings in the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        This data is split based on the subject ID
        type: pandas DataFrame
        
        Returns:
        Two values for daytime and nighttime mean averaged out over the days in the individuals readings
        """
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
    

    def smoothing(self, data):
        """
            This method performs a EWM smoothening for a time-series to improve MAGE calculation
            
            Function Parameters:
            data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package and the Ohio University with the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            This data is split based on the subject ID
            type: pandas DataFrame

            Returns:
            A pandas DataFrame with smoothened glucose values
        """ 
        
        data.GlucoseValue = data.GlucoseValue.ewm(alpha=0.05, adjust=False).mean()
        data.GlucoseValue = data.GlucoseValue.round()
        
        return data


    def subSample(self, data):
        """
            This function subsamples a time-series at 15 minute intervals

            Function Parameters:

            data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package and the Ohio University with the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            This data is split based on the subject ID
            type: pandas DataFrame

            Returns:
            A pandas DataFrame with reading at 15 minute intervals


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
            This function converts glucose values into the desired unit of measurement

            Function Parameters:
            data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package and the Ohio University with the following format:
            Display Time     object
            GlucoseValue    float64
            subjectId        object
            This data is split based on the subject ID
            type: pandas DataFrame
            
            units   It is the desired unit of conversion    type: string

            Returns:
            The individual time series desired unit of glucose value
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
        dates = list()
        data = data.reset_index(drop=True)
        for i in range(0,len(data.index)):
            dates.append(data['Display Time'][i].date())
        data['Dates'] = dates
        end = data['Dates'].iloc[-1]
        start = data['Dates'].iloc[0]

        indexVals = data[ data['Dates'] == start ].index
        data.drop(indexVals , inplace=True)

        indexVals = data[ data['Dates'] == end ].index
        data.drop(indexVals , inplace=True)

        data = data.reset_index(drop=True)
        data.drop(['Dates'], axis=1, inplace=True)
        
        return data


    def fullDaysOnly(self, data):
        """
        Function for trimming an individual's glucose values to only consist of full days

        Function Parameters:
        data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package and the Ohio University with the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        This data is split based on the subject ID
        type: pandas DataFrame

        Returns:
        The individuals time series with data only for full days
        """
        data_fullDays = pd.DataFrame()

        for subjectId, df in data.groupby('subjectId'):
            df['Display Time'] = pd.to_datetime(df['Display Time'])
            df = df.reset_index(drop=True)
            temp = self.fullDay(df)
            data_fullDays = pd.concat([data_fullDays, temp],ignore_index=True)

        return(data_fullDays)


    def hourlySamples(self, df):
        """
        This function subsamples a time-series at 1-hour intervals

        Function Parameters:
        data: The default dataset (CSV file) includes data from the CGMAnalysis package, Gluvarpro package, CGMAnalyzer package and the Ohio University with the following format:
        Display Time     object
        GlucoseValue    float64
        subjectId        object
        This data is split based on the subject ID
        type: pandas DataFrame

        Returns:
        A pandas DataFrame with reading at 1-hour intervals
        """
        groupkey = df['Display Time'].values.astype('datetime64[h]')
        result = df.groupby(groupkey).first()
        result = result.reset_index(drop=True)
        return (result)


        
#==================================================================================================================
#   These methods are not called by the user and only used for internal processing
#==================================================================================================================
    
   

    def detectGap(self, testing_data):
        l = []
        k = -1
        for i in testing_data['GlucoseValue']:
            k+=1
            if math.isnan(i):
                l.append(k)
        b = min(l)
        e = max(l)

        gap=e-b

        return b,e,gap



#==================================================================================================================
#   Calculating error metrics
#==================================================================================================================

    def index_agreement(self, s,o):
        """
        This function calculates the Index of Agreement of the imputed values with respect to the original values

        Function Parameters:
        s   a list of the predicted values  type: NumPy array
        o   a list of the original values   type: NumPy array
        
        Return:
        A numerical value of the index of agreement
        """

        # ioa = 1 - [ ( sum( (obs - sim)^2 ) ] / sum( ( abs(sim - mean(obs)) + abs(obs - mean(obs)) )^2 )
        
        ia = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
        return ia

    
    def rmse(self, s,o):
        """
        This function calculates the Root Mean Squared Error of the imputed values with respect to the original values

        Function Parameters:
        s   a list of the predicted values  type: NumPy array
        o   a list of the original values   type: NumPy array

        Return:
        A numerical value of the root mean squared error

        """
        return np.sqrt(np.mean((s-o)**2))

    
    def mad(self, s,o):
        """
        This function calculates the Mean Absolute Deviationrrorof the imputed values with respect to the original values

        Function Parameters:
        s   a list of the predicted values  type: NumPy array
        o   a list of the original values   type: NumPy array

        Return:
        A numerical value of the mean absolute difference

        """
        return np.mean(abs(s-o))

    
    def mape(self, y_pred,y_true):
        """
        This function calculates the Mean Absolute Percentage Error of the imputed values with respect to the original values

        Function Parameters:
        y_pred  a list of the predicted values  type: NumPy array
        y_true  a list of the original values   type: NumPy array

        Return:
        A numerical value of the Mean Absolute Percentage error
        """
    
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    def fracBias(self, s,o):
        """
        This function calculates the Fractional Bias of the imputed values with respect to the original values
        
        Function Parameters:
        s   a list of the predicted values  type: NumPy array
        o   a list of the original values   type: NumPy array

        Return:
        A numerical value of the Fractional Bias

        """
        
        return np.mean(np.abs((o - s) / ((o + s)/2)))

    
    def getMetrics(self,lstm_pred, test_val):
        """
        This function is a wrapper for all the error metrics

        lstm_pred   a list of the predicted values  type: NumPy array
        test_val    a list of the original values   type: NumPy array
       
        Return:
        Outputs the error metrics
        """
        #IOA
        ioa_val = self.index_agreement(lstm_pred,test_val)
        print("Index of Agreement is: " + str(round(ioa_val,3)))
    
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
 

