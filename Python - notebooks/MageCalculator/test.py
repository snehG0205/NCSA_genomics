from mage_calc import MageDataSet
times = [60*i for i in range(1000)]
'''glucs = [200, 180, 160, 140, 120, 100, 70, 114, 80, 95, 120, 140, 160, \
140, 100, 150, 170, 220, 215, 210, 170, 140, 200, 60, 65, 75, 85, 95, \
140, 80, 60, 80, 100, 120, 160, 180, 240, 220, 170, 250, 300, 240, 200, \
150, 125, 100, 140, 180]
'''
glucs = [100, 150, 100, 130, 100, 150, 100, 130, 100, 150, 100, 130, \
        100, 150, 100, 130, 100, 150, 100, 130, 100, 150, 100, 130]
m = MageDataSet(times, glucs)#[0, 500, 1400, 2000, 2400],[200,250,120, 180, 160])
m.printEverything(True)
m.getMAGE()
print(m)