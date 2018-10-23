import matplotlib.pyplot as plt
import numpy as np

fontTitle = {
        'family': 'fantasy',
        'color':  '#FFD700',
        'weight': 'demibold',
        'size': 14
        }

fontHead = {
        'family': 'sans-serif',
        'color':  '#FFD700',
        'weight': 'demibold',
        'size': 30
}

fontY = {
        'family': 'sans-serif',
        'color':  '#FFD700',
        'weight': 'normal',
        'size': 9,
        }

fontNum = {
    'family': 'monospace',
    'color': 'black',
    'weight': 'roman',
    'stretch': 'ultra-expanded',
    'size': 6.5
    }

lebron_info = open("D:/SparkProjects/NBADataBase/data/lebron_data.csv")
yearList = [int(i) for i in lebron_info.readline().split(',')]
pointsList = [float(i) for i in lebron_info.readline().split(',')]
reboundsList = [float(i) for i in lebron_info.readline().split(',')]
assistsList = [float(i) for i in lebron_info.readline().split(',')]
stealsList = [float(i) for i in lebron_info.readline().split(',')]
blocksList = [float(i) for i in lebron_info.readline().split(',')]
gamesList = [float(i) for i in lebron_info.readline().split(',')]
fieldPercentList = [float(i) for i in lebron_info.readline().split(',')]
threePercentList = [float(i) for i in lebron_info.readline().split(',')]

fig = plt.figure(figsize=[20, 10], facecolor='#CD00CD')
fig.suptitle("LeBron James's Statistics 2003-2018", fontstyle='italic',
             fontweight='bold', fontsize=30, color='#FFD700',
             horizontalalignment='center', verticalalignment='top')
axs = fig.subplots(2, 4)

axs[0][0].set_title('Average Points', fontdict=fontTitle)
axs[0][0].bar(yearList, pointsList, hatch='+', color='#FFD700', edgecolor='#8B0A50')
axs[0][0].set_yticks(np.arange(0, 35, 3))
axs[0][0].set_xlim(2003, 2018)
for a, b in zip(yearList, pointsList):
    axs[0][0].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[0][1].set_title('Average Rebounds', fontdict=fontTitle)
axs[0][1].bar(yearList, reboundsList, hatch='O', color='#FFD700', edgecolor='#8B0A50')
axs[0][1].set_yticks(np.arange(0, 9, 1))
axs[0][1].set_xlim(2003, 2018)
for a, b in zip(yearList, reboundsList):
    axs[0][1].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[0][2].set_title("Average Assists", fontdict=fontTitle)
axs[0][2].bar(yearList, assistsList, hatch='|', color='#FFD700', edgecolor='#8B0A50')
axs[0][2].set_yticks(np.arange(0, 9, 1))
axs[0][2].set_xlim(2003, 2018)
for a, b in zip(yearList, assistsList):
    axs[0][2].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[0][3].set_title("Average Steals", fontdict=fontTitle)
axs[0][3].bar(yearList, stealsList, hatch='X', color='#FFD700', edgecolor='#8B0A50')
axs[0][3].set_yticks(np.arange(0, 2.5, 0.25))
axs[0][3].set_xlim(2003, 2018)
for a, b in zip(yearList, stealsList):
    axs[0][3].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[1][0].set_title("Average Blocks", fontdict=fontTitle)
axs[1][0].bar(yearList, blocksList, hatch='*', color='#FFD700', edgecolor='#8B0A50')
axs[1][0].set_yticks(np.arange(0, 1.4, 0.1))
axs[1][0].set_xlim(2003, 2018)
for a, b in zip(yearList, blocksList):
    axs[1][0].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[1][1].set_title("Field Goals Percent", fontdict=fontTitle)
axs[1][1].bar(yearList, fieldPercentList, hatch='.', color='#FFD700', edgecolor='#8B0A50')
axs[1][1].set_yticks(np.arange(0, 0.7, 0.05))
axs[1][1].set_xlim(2003, 2018)
for a, b in zip(yearList, fieldPercentList):
    axs[1][1].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[1][2].set_title("ThreePoint Goals Percent", fontdict=fontTitle)
axs[1][2].bar(yearList, threePercentList, hatch='/', color='#FFD700', edgecolor='#8B0A50')
axs[1][2].set_yticks(np.arange(0, 0.5, 0.05))
axs[1][2].set_xlim(2003, 2018)
for a, b in zip(yearList, threePercentList):
    axs[1][2].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)

axs[1][3].set_title("Games Attended", fontdict=fontTitle)
axs[1][3].bar(yearList, gamesList, hatch='-', color='#FFD700', edgecolor='#8B0A50')
axs[1][3].set_yticks(np.arange(0, 85, 5))
axs[1][3].set_xlim(2003, 2018)
for a, b in zip(yearList, gamesList):
    axs[1][3].text(a, b, b,
                   horizontalalignment='center', verticalalignment='baseline',
                   fontdict=fontNum)
plt.savefig('D:/SparkProjects/NBADataBase/data/lebron_statsfig.png', facecolor='#CD00CD')
plt.show()
