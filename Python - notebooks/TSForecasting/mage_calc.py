import operator  # Used to sort by an element of a class
import collections  # Used to get a dictionary with .append()
try:
    from enum import Enum  # Used to make algorithma bit more readable
    class PointType(Enum):
        Nothing = -1
        Peak = 0
        Valley = 1
except ImportError:
    class PointType(object):
        Nothing = -1
        Peak = 0
        Valley = 1
try:
    from numpy import std
except ImportError:
    import math
    def std(ll):
        n = float(len(ll))
        if n < 1:
            raise ValueError('Mean requires at least one data point!')
        avg = float(0)
        for el in ll:
            avg += el
        avg /= n
        sigma_sq = float(0)
        for el in ll:
            sigma_sq += (el - avg)**2
        sigma_sq /= n
        return math.sqrt(sigma_sq)
class MageDataPoint(object):
    '''

    This class defines a single data point,
    consisting of a time in minutes, glucose
    value in mg/dL, and a standard deviation in
    mg/dL. Additionally, we define useful operations
    on the class, like -, >, >=, <, <=, and print()

    '''

    def __init__(self, newTime=-1, newGlucose=-1):
        if newTime == -1 or newGlucose == -1:
            raise ValueError('Mage Data Points need both time and glucose!') 
        self.t = newTime
        self.g = newGlucose
        self.stdev = 0

    def gluc(self):
        # Access glucose value of the current object
        return self.g

    def __str__(self):
        return ""
        # return "Indice: {},\tGlucose: {},\tStDev: {}"\
        #         .format(self.t, self.g, self.stdev)

    def plusSigma(self):
        return self.g + self.stdev

    def minusSigma(self):
        return self.g - self.stdev

    def __sub__(self, other):
        if isinstance(other, MageDataPoint):
            return self.g - other.g
        elif isinstance(other, (int, float)):
            return self.g - other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, MageDataPoint):
            return self.g < other.g
        elif isinstance(other, (int, float)):
            return self.g < other
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, MageDataPoint):
            return self.g <= other.g
        elif isinstance(other, (int, float)):
            return self.g <= other
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MageDataPoint):
            return self.g > other.g
        elif isinstance(other, (int, float)):
            return self.g > other
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MageDataPoint):
            return self.g >= other.g
        elif isinstance(other, (int, float)):
            return self.g >= other
        else:
            return NotImplemented


class MageDataSet(object):
    '''

    This class defines a dataset. The default constructor
    takes as its argument a list of times and glucose values.
    The .calculate() method can be used to calculate MAGE, and
    the getMAGE() method can be used to get the most recently
    calculated MAGE without redoing the entire calculation.

    '''

    def __init__(self, newTimes=None, newGlucoses=None):
        self.OneDay = 1440  # length of a day in minutes
        self.MAGE = -1
        self.NUM_MAGE_PTS = 1
        self.printPoints = False
        if newTimes is None or newGlucoses is None:
            self.points = list()
        else:
            '''
            Make sure you dont overstep your bounds when indexing
            the incoming arrays, because they may be of unequal sizes
            '''
            self.points = [MageDataPoint(newTimes[i], newGlucoses[i])
                           for i in range(min(len(newTimes),
                                len(newGlucoses)))]

    def __str__(self):
        if self.printPoints:
            for point in self.points:
                print(point)
        return "".format(self.getMAGE())

    def printEverything(self, booly):
        if isinstance(booly, bool):
            self.printPoints = booly

    def sortByTime(self):
        self.points.sort(key=operator.attrgetter('t'))

    def prepareValues(self):
        self.sortByTime()
        self.pointsByDay = collections.defaultdict(list)
        # Group all points by day, using a dictionary
        for point in self.points:
            self.pointsByDay[point.t // self.OneDay].append(point)
        '''
        Once grouped, calculate the standard deviation
        for that day, and set the corresponding point in the
        actual object up with that standard deviation
        '''
        daily_offset = 0
        for day, dailyData in self.pointsByDay.items():
            stddev = std([point.gluc() for point in dailyData])
            for offset, point in enumerate(dailyData):
                point.stdev = stddev
                self.points[day * daily_offset + offset] = point
            daily_offset = len(dailyData)

    def current(self):
        if self.pointIndex >= 0:
            return self.points[self.pointIndex]
        else:
            return self.points[0]

    def findFirstPeakAndValley(self):
        '''
        Can't even start until we have 3 points.
        Makes sense, since MAGE is a Peak finding
        algorithm at its core.
        '''
        if len(self.points) < 3:
            return
        self.pointIndex = 1
        found = False
        self.lastFound = PointType.Nothing
        '''
        Forgive me, this next bit is O(n^2)
        '''
        while(found is False):
            for point in self.points[:self.pointIndex]:
                if self.current() >= point.plusSigma():
                    found = True
                    self.lastFound = PointType.Peak
                    self.currentPeak = self.current()
                    self.currentValley = point
                    break  #return#break
                elif self.current() <= point.minusSigma():
                    found = True
                    self.lastFound = PointType.Valley
                    self.currentPeak = point
                    self.currentValley = self.current()
                    break  #return#break
            self.pointIndex += 1

    def findOtherPeaksAndValleys(self):
        '''
        again, we constrain the number
        of points since it doesn't make
        sense to move on if the algorithm
        doesn't have enough data.
        '''
        if len(self.points) < 3:
            return
        self.mage = 0
        self.num_mage_pts = 0
        for point in self.points[self.pointIndex:]:
            '''
            print("pp: {}, vv: {}, cc: {}".format\
            (self.currentPeak, self.currentValley, point))
            '''
            if self.lastFound == PointType.Valley:
                if point >= self.currentValley.plusSigma():
                    '''
                    we found a Peak! now safe the add the previous two
                    to our running mage sum
                    '''
                    # print("found Peak!\t", point)
                    self.lastFound = PointType.Peak
                    self.mage += abs(self.currentPeak - self.currentValley)
                    self.num_mage_pts += 1
                    self.currentPeak = point
                else:
                # check for smaller Valley
                    #print("checking for smaller Valley,\t", self.currentValley)
                    self.currentValley = point if point < self.currentValley else self.currentValley
                    #print("maybe new Valley?\t\t\t", self.currentValley)
            elif self.lastFound == PointType.Peak:
                if point <= self.currentPeak.minusSigma():
                    '''
                    we found a Valley! now safe to add the previous two
                    to our running mage sum
                    '''
                    # print("found Valley!\t", point)
                    self.lastFound = PointType.Valley
                    self.mage += abs(self.currentPeak - self.currentValley)
                    self.num_mage_pts += 1
                    self.currentValley = point
                else:
                    #print("checking for larger Peak,\t", self.currentPeak)
                    self.currentPeak = point if point > self.currentPeak else self.currentPeak
                    #print("maybe new Peak?\t\t\t", self.currentPeak)
            else:
                raise ValueError('Attempted to find next peak/valley\
                                 without an initial! Exiting.') 
        top = self.currentPeak
        bottom = self.currentValley
        if self.lastFound == PointType.Peak:
            #print("last found was a Peak")
            if point <= self.currentPeak.minusSigma():
                top = point
            else:
                top = self.currentValley
            bottom = self.currentPeak
        elif self.lastFound == PointType.Valley:
            #print("last found was a Valley")
            if point >= self.currentValley.plusSigma():
                top = point
            else:
                top = self.currentPeak
            bottom = self.currentValley
        self.mage += abs(top - bottom)
        self.num_mage_pts += 1
            # self.pointIndex += 1 # do this so we can use .current() properly

    def getMAGE(self):
        if(self.MAGE < 0):
            self.calculate()
        return str(self.MAGE/self.NUM_MAGE_PTS)

    def calculate(self):
        self.prepareValues()
        self.findFirstPeakAndValley()
        self.findOtherPeaksAndValleys()
        
