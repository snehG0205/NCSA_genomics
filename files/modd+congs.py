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
        return round(mean(Modd),2) 



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