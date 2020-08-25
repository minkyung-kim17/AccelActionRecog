import pdb
import os
import numpy as np
import matplotlib.pyplot as plt

action_dic = {'5': 'Throw',
			  '7': 'Basketball shoot',
			  '12': 'Bowling',
			  '15': 'Tennis swing',
              '16': 'Arm curl',
			  '18': 'Push',
			  '21': 'Pickup and throw',
			  '23': 'Walk'}

# files = [x for x in os.listdir(os.getcwd()) if x[-3:] == 'npy']
files = [x for x in os.listdir('./NPY') if x[-3:] == 'npy']

for actionNum in action_dic.keys():
    fig = plt.figure(figsize=(30, 10))
    plt.title('{}'.format(action_dic[actionNum]))

    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    for subject in range(1, 8+1):
        actionFiles = [x for x in files if x.split('_')[0][1:] == actionNum and x.split('_')[1][1:] == str(subject)]

        for file in actionFiles:
            data = np.load(os.path.join('./NPY', file))

            ax = data[:, 0]
            ay = data[:, 1]
            az = data[:, 2]

            a_tot = np.sqrt(ax**2 + ay**2 + az**2)

            axes[subject-1].plot(a_tot)

    # plt.savefig('{}.png'.format(action_dic[actionNum]))

plt.show()
