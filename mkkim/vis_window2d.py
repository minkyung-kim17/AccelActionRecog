import pdb
import os
import numpy as np
import matplotlib.pyplot as plt

# 사람별로, 액션 마다 다른 색으로, 윈도우 별 mean, std로 2d scatter plot을 하고자 함. 

action_dic = {'5': 'Throw',
			  '7': 'Basketball shoot',
			  '12': 'Bowling',
			  '15': 'Tennis swing',
			  '18': 'Push',
			  '21': 'Pickup and throw'}
			  # '23': 'Walk'}

# recFreq = 50
windowSize = 50  # 0.5s
windowInterval = 10

colorset = {'5': 'r',
            '7': 'g',
            '12': 'b',
            '15': 'k',
            '18': 'm',
            '21': 'y'}

files = [x for x in os.listdir('./NPY') if x[-3:] == 'npy']

fig = plt.figure(figsize=(30, 10))
plt.title('WindowSize ({}), windowInterval ({})'.format(windowSize, windowInterval))

ax1 = fig.add_subplot(241)
ax2 = fig.add_subplot(242)
ax3 = fig.add_subplot(243)
ax4 = fig.add_subplot(244)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

subject_minmax = {}

for subject in range(1, 8+1):
    # plt.figure(subject)

    for actionNum in sorted(action_dic.keys()):
        actionDataWin = []

        actionFiles = [x for x in files if x.split('_')[0][1:] == actionNum and x.split('_')[1][1:] == str(subject)]

        for file in actionFiles:
            data = np.load(os.path.join('./NPY', file))

            ax = data[:, 0]
            ay = data[:, 1]
            az = data[:, 2]

            a_tot = np.sqrt(ax**2 + ay**2 + az**2)

            # print(len(a_tot))
            action = []
            for i in range(0, len(a_tot), windowInterval):
                if i+windowSize <= len(a_tot):
                    action.append(a_tot[i:i+windowSize])

            action = np.array(action)
            # pdb.set_trace()

            actionDataWin.append(action)

        actionDataWin = np.array(actionDataWin)

        for i in range(len(actionDataWin)):
            windows = actionDataWin[i]
            windowMS = np.vstack((np.mean(windows, axis=1), np.std(windows, axis=1))).T

            if i + 1 == len(actionDataWin):
                axes[subject-1].plot(windowMS[:, 0], windowMS[:, 1], '.--', label='{}'.format(action_dic[actionNum]), color=colorset[actionNum])
            else:
                axes[subject-1].plot(windowMS[:, 0], windowMS[:, 1], '.--', color=colorset[actionNum])

    axes[subject-1].legend()
plt.show()

