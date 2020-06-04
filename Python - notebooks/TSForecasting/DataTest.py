import pandas as pd
import numpy as np

import os

class DataTest:
	cwd = os.getcwd()
	consolidated_paper = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPaper.csv')
	consolidated_pkg = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPackage.csv')
	def_training = pd.read_csv(cwd+'/TSForecasting/Data/consolidatedDataForPaper.csv')

	def showData(self):
		display(consolidated_paper)
		display(consolidated_pkg)
		display(def_training)

	def modifyData(self):
		self.consolidated_paper = self.fullDaysOnly(self.consolidated_paper)
		self.consolidated_pkg = self.fullDaysOnly(self.consolidated_pkg)
		self.def_training = self.fullDaysOnly(self.def_training)

	def fullDay(self, data):
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

		return data_fullDays




