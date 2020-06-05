import operator  # Used to sort by an element of a class
from enum import Enum  # Used to make algorithma bit more readable
import collections  # Used to get a dictionary with .append()
import numpy  # Used to calculate standard deviation


class PointType(Enum):
    Nothing = -1
    Peak = 0
    Valley = 1


class MageDataPoint(object):
    '''
    This class defines a single data point,
    consisting of a time in minutes, glucose
    value in mg/dL, and a standard deviation in
    mg/dL. Additionally, we define useful operations
    on the class, like -, >, >=, <, <=, and print()
    '''

    def __init__(self, newTime=-1, newGlucose=-1, s_dev=5):
        if(newTime == -1 or newGlucose == -1):
            raise RuntimeError
        self.t = newTime
        self.g = newGlucose
        self.stdev = s_dev

    def gluc(self):
        # Access glucose value of the current object
        return self.g

    def __str__(self):
        return "Time: {},\tGlucose: {},\tStDev: {}"\
                .format(self.t, self.g, self.stdev)

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

    def __init__(self, newTimes=None, newGlucoses=None, s_dev=5):
        self.OneDay = 1440  # length of a day in minutes
        self.MAGE = -1
        self.NUM_MAGE_PTS = 1
        self.printPoints = False
        self.stdev = s_dev
        if(newTimes is None or newGlucoses is None):
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
        if(self.printPoints):
            for point in self.points:
                print(point)
        return "MAGE score: {}".format(self.getMAGE())

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
            std = numpy.std([point.gluc() for point in dailyData])
            for offset, point in enumerate(dailyData):
                point.stdev = std
                self.points[day * daily_offset + offset] = point
            daily_offset = len(dailyData)

    def current(self):
        if(self.pointIndex >= 0):
            return self.points[self.pointIndex]
        else:
            return self.points[0]

    def findFirstPeakAndValley(self):
        '''
        Can't even start until we have 3 points.
        Makes sense, since MAGE is a peak finding
        algorithm at its core.
        '''
        if(len(self.points) < 3):
            return
        self.pointIndex = 1
        found = False
        self.lastFound = PointType.Nothing
        while(found is False):
            '''
            Forgive me, this next bit is O(n^2)
            '''
            for point in self.points[:self.pointIndex]:
                if(self.current() >= point.plusSigma()):
                    found = True
                    self.lastFound = PointType.Peak
                    self.currentPeak = self.current()
                    self.currentValley = point
                    break  #return#break
                elif(self.current() <= point.minusSigma()):
                    found = True
                    self.lastFound = PointType.Valley
                    self.currentPeak = point
                    self.currentValley = self.current()
                    break  #return#break
            self.pointIndex += 1

    def findOtherPeaksAndValleys(self):
        '''
        Again, we constrain the number
        of points since it doesn't make
        sense to move on if the algorithm
        doesn't have enough data.
        '''
        if(len(self.points) < 3):
            return
        self.MAGE = 0
        self.NUM_MAGE_PTS = 0
        for point in self.points[self.pointIndex:]:
            
            # print("pp: {}, vv: {}, cc: {} \n".format\
            # (self.currentPeak, self.currentValley, point))

            if(self.lastFound == PointType.Valley):
                if(point >= self.currentValley.plusSigma()):
                    '''
                    We found a peak! Now safe the add the previous two
                    to our running MAGE sum
                    '''
                    # print("\nfound peak!\t", point)
                    self.lastFound = PointType.Peak
                    self.MAGE += abs(self.currentPeak - self.currentValley)
                    self.NUM_MAGE_PTS += 1
                    self.currentPeak = point
                else:
                # Check for smaller valley
                    # print("\nchecking for smaller valley,\t", self.currentValley)
                    self.currentValley = point if point < self.currentValley else self.currentValley
                    # print("\nmaybe new valley?\t\t\t", self.currentValley)
            elif(self.lastFound == PointType.Peak):
                if(point <= self.currentPeak.minusSigma()):
                    '''
                    We found a valley! Now safe to add the previous two
                    to our running MAGE sum
                    '''
                    # print("\nfound valley!\t", point)
                    self.lastFound = PointType.Valley
                    self.MAGE += abs(self.currentPeak - self.currentValley)
                    self.NUM_MAGE_PTS += 1
                    self.currentValley = point
                else:
                    # print("\nchecking for larger peak,\t", self.currentPeak)
                    self.currentPeak = point if point > self.currentPeak else self.currentPeak
                    # print("\nmaybe new peak?\t\t\t", self.currentPeak)
            else:
                print("Uhoh, something's gone terribly wrong!")
                print("Somehow we're filtering when lastFound is Nothing!")
                raise RuntimeException
        top = self.currentPeak
        bottom = self.currentValley
        if self.lastFound == PointType.Peak:
            # print("\nlast found was a peak\n")
            if(point <= self.currentPeak.minusSigma()):
                top = point
            else:
                top = self.currentValley
            bottom = self.currentPeak
        elif self.lastFound == PointType.Valley:
            # print("\nlast found was a valley\n")
            if(point >= self.currentValley.plusSigma()):
                top = point
            else:
                top = self.currentPeak
            bottom = self.currentValley
        self.MAGE += abs(top - bottom)
        self.NUM_MAGE_PTS += 1
            # self.pointIndex += 1 # Do this so we can use .current() properly

    def getMAGE(self):
        if self.MAGE < 0:
            self.calculate()
        return self.MAGE/self.NUM_MAGE_PTS

    def calculate(self):
        self.prepareValues()
        self.findFirstPeakAndValley()
        self.findOtherPeaksAndValleys()
        return self.getMAGE()

# if __name__ == '__main__':
#     times = [60*i for i in range(1000)]
#     '''glucs = [200, 180, 160, 140, 120, 100, 70, 114, 80, 95, 120, 140, 160, \
#     140, 100, 150, 170, 220, 215, 210, 170, 140, 200, 60, 65, 75, 85, 95, \
#     140, 80, 60, 80, 100, 120, 160, 180, 240, 220, 170, 250, 300, 240, 200, \
#     150, 125, 100, 140, 180]
#     '''
#     glucs = [100, 150, 100, 130, 100, 150, 100, 130, 100, 150, 100, 130, \
#             100, 150, 100, 130, 100, 150, 100, 130, 100, 150, 100, 130]
#     m = MageDataSet(times, glucs)#[0, 500, 1400, 2000, 2400],[200,250,120, 180, 160])
#     m.printEverything(True)
#     m.getMAGE()
#     print(m)