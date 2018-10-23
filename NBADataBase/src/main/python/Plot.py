import matplotlib.pyplot as plt
import numpy as np

arrowDict = {
    'arrowstyle': 'fancy'
}

plt.style.use('ggplot')

lebron_info = open("D:/SparkProjects/NBADataBase/data/lebron_data.csv")
lineInfo = lebron_info.readline().split(',')
yearList = [int(i) for i in lineInfo[0:14]]
pointsList = [float(i) for i in lineInfo[14:28]]
reboundsList = [float(i) for i in lineInfo[28:42]]
assistsList = [float(i) for i in lineInfo[42:56]]

plt.plot(yearList, pointsList, 'r>-', label='points')
plt.plot(yearList, reboundsList, 'bx-.', label='rebounds')
plt.plot(yearList, assistsList, 'cD--', label='assists')
plt.annotate('Rookie Year!', (2004, 20.937), xytext=(2005, 17),
             arrowprops=arrowDict)

plt.xlabel('Year')
plt.legend(loc=7, fancybox=True, shadow=True, title='Info Tags')

plt.show()