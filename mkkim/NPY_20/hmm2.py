import pdb
import os
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import argparse
from random import *

action_dic = {
			  '5': 'Throw',
			  '7': 'Basketball shoot',
			  '12': 'Bowling',
			  '15': 'Tennis swing',
			  # '16': 'Arm curl',
			  # '18': 'Push',
			  # '21': 'Pickup and throw',  # pickup은 normal이랑 겹치고, throw는 throw랑 겹침
			  '100': 'Normal',
			  '200': 'Abnormal'}


windowSize = 20  # 1s
windowInterval = 4
numStates = 12

trte_rate = 0.80  # 0.95로 실험했었음
AllFiles = [x for x in os.listdir('./') if x[-3:] == 'npy' and x.split('_')[0][1:] in action_dic.keys()]
trte_idx = np.random.permutation(len(AllFiles))
AllFiles_tr = np.array(AllFiles)[trte_idx[:int(len(AllFiles)*trte_rate)]]
AllFiles_te = np.array(AllFiles)[trte_idx[int(len(AllFiles)*trte_rate):]]

# 한 사람만 제외
parser = argparse.ArgumentParser()
parser.add_argument('--testSubject', '-s')
args = parser.parse_args()

if args.testSubject is not None:
	test_subject = int(args.testSubject)
	AllFiles_tr = [x for x in os.listdir(os.getcwd()) if x[-3:] == 'npy' and int(x.split('_')[1][1:]) != test_subject and x.split('_')[0][1:] in action_dic.keys()]
	AllFiles_te = [x for x in os.listdir(os.getcwd()) if x[-3:] == 'npy' and int(x.split('_')[1][1:]) == test_subject and x.split('_')[0][1:] in action_dic.keys()]

print(' ')
print('{} Files'.format(len(AllFiles)))
print('Training: {}, Test: {}'.format(len(AllFiles_tr), len(AllFiles_te)))
print(' ')

## Data Preprocessing
def getWindowData(actionFileList):
	data_tr = {}
	for actionNum in action_dic.keys():
		actionFiles = [x for x in actionFileList if x.split('_')[0][1:] == actionNum]

		actionDataWinAll = []  # action type 하나에 대해서, 그 안에 있는 각 개별 action에 대한 windowed data가 있음
		for file in actionFiles:
			data = np.load(os.path.join('./', file))

			ax = data[:, 0]
			ay = data[:, 1]
			az = data[:, 2]

			gx = data[:, 3]
			gy = data[:, 4]
			gz = data[:, 5]

			a_tot = np.sqrt(ax**2 + ay**2 + az**2)  # approach 1
			# a_tot = np.vstack((ax, ay, az, gx, gy, gz))  # (6, length)  # approach 2

			actionDataWin = []  # action 하나를 window로 잘랐을 때, 모든 window를 matrix 형태로 묶음
			for i in range(0, len(a_tot), windowInterval):
				if i+windowSize <= len(a_tot):
					actionDataWin.append(a_tot[i:i+windowSize])
			actionDataWinAll.append(np.array(actionDataWin))

			# actionDataWin = []  # action 하나를 window로 잘랐을 때, 모든 window를 matrix 형태로 묶음
			# for i in range(0, a_tot.shape[1], windowInterval):
			#     if i+windowSize <= a_tot.shape[1]:
			#         actionDataWin.append(a_tot[:, i:i+windowSize])
			# actionDataWinAll.append(np.array(actionDataWin))

		actionDataWinAll = np.array(actionDataWinAll)
		# pdb.set_trace()  # approach 2로 바꾸는 작업하는 중!!

		actionDataWinAllMS = []  # convert to (mean, std)  # approach 1
		for i in range(len(actionDataWinAll)):
			actionDataWin = actionDataWinAll[i]
			actionDataWinAllMS.append(np.vstack((np.mean(actionDataWin, axis=1), np.std(actionDataWin, axis=1))).T)
		actionDataWinAllMS = np.array(actionDataWinAllMS)

		data_tr[actionNum] = actionDataWinAllMS

	return data_tr

data_tr = getWindowData(AllFiles_tr)
data_te = getWindowData(AllFiles_te)

# data_tr = {}
# for actionNum in action_dic.keys():
#     actionFiles = [x for x in AllFiles_tr if x.split('_')[0][1:] == actionNum]

#     actionDataWinAll = []  # action type 하나에 대해서, 그 안에 있는 각 개별 action에 대한 windowed data가 있음
#     for file in actionFiles:
#         data = np.load(os.path.join('./', file))

#         ax = data[:, 0]
#         ay = data[:, 1]
#         az = data[:, 2]

#         gx = data[:, 3]
#         gy = data[:, 4]
#         gz = data[:, 5]

#         a_tot = np.sqrt(ax**2 + ay**2 + az**2)  # approach 1
#         # a_tot = np.vstack((ax, ay, az, gx, gy, gz))  # (6, length)  # approach 2

#         actionDataWin = []  # action 하나를 window로 잘랐을 때, 모든 window를 matrix 형태로 묶음
#         for i in range(0, len(a_tot), windowInterval):
#             if i+windowSize <= len(a_tot):
#                 actionDataWin.append(a_tot[i:i+windowSize])
#         actionDataWinAll.append(np.array(actionDataWin))

#         # actionDataWin = []  # action 하나를 window로 잘랐을 때, 모든 window를 matrix 형태로 묶음
#         # for i in range(0, a_tot.shape[1], windowInterval):
#         #     if i+windowSize <= a_tot.shape[1]:
#         #         actionDataWin.append(a_tot[:, i:i+windowSize])
#         # actionDataWinAll.append(np.array(actionDataWin))

#     actionDataWinAll = np.array(actionDataWinAll)
#     # pdb.set_trace()  # approach 2로 바꾸는 작업하는 중!!

#     actionDataWinAllMS = []  # convert to (mean, std)  # approach 1
#     for i in range(len(actionDataWinAll)):
#         actionDataWin = actionDataWinAll[i]
#         actionDataWinAllMS.append(np.vstack((np.mean(actionDataWin, axis=1), np.std(actionDataWin, axis=1))).T)
#     actionDataWinAllMS = np.array(actionDataWinAllMS)

#     data_tr[actionNum] = actionDataWinAllMS

## HMM modeling  --> action 전체를 모아서 modeling
data = np.random.random((0, 2))  # (mean, std)
datalen = []
for actionNum in action_dic.keys():
	for i in range(len(data_tr[actionNum])):
		dataWin = data_tr[actionNum][i]
		datalen.append(len(dataWin))
		data = np.concatenate([data, dataWin])
model = hmm.GaussianHMM(n_components=numStates).fit(data, datalen)

## class별 hidden state 조사
def getHiddens(windowedData):
	hiddens = {}
	for actionNum in action_dic.keys():
		hiddens[actionNum] = []
		for i in range(len(windowedData[actionNum])):
			actionData = windowedData[actionNum][i]
			hidden = model.predict(actionData)
			hiddens[actionNum].append(hidden)
	return hiddens

hiddens_tr = getHiddens(data_tr)
hiddens_te = getHiddens(data_te)

# pdb.set_trace()

## class별 hidden state의 flow visualization
ran = 0.1
statePoints = np.array([[2, 12],
						[2, 9],
						[2, 5],
						[2, 1],
						[5, 12],
						[5, 9],
						[5, 5],
						[5, 1],
						[8, 12],
						[8, 9],
						[8, 5],
						[8, 1]])
for actionNum in action_dic.keys():
	plt.figure()
	plt.title(action_dic[actionNum])
	for i, statePoint in enumerate(statePoints):
		plt.plot(statePoint[0], statePoint[1], 'k.', markerSize=30, alpha=0.1)
		plt.text(statePoint[0]-0.2, statePoint[1]-0.2, str(i))
	plt.xlim(0, 10)
	plt.ylim(0, 13)

	for states in hiddens_tr[actionNum]:

		tempX = [statePoints[x][0]+uniform(-ran, ran) for x in states]
		tempY = [statePoints[x][1]+uniform(-ran, ran) for x in states]

		plt.plot(tempX[0], tempY[0], 'rx')

		for i in range(len(tempX)-1):
			px1 = tempX[i]
			py1 = tempY[i]

			px2 = tempX[i+1]
			py2 = tempY[i+1]

			plt.arrow(px1, py1,	px2-px1, py2-py1, head_width=0.01)
			plt.plot(px2, py2, 'b.')  # 도착 state

		plt.plot(tempX[-1], tempY[-1], 'bx')

# plt.show()
# pdb.set_trace()

# normalC에 대한 transitionProbability
normalC = {
		   # '16': 'Arm curl',
		   # '18': 'Push',
		   '100': 'Normal'
		  }
normalC_transmat_ = np.zeros((numStates, numStates))

for actionNum in action_dic.keys():
	if actionNum not in normalC:
		continue

	cnt_total = 0
	for states in hiddens_tr[actionNum]:
		for i, j in zip(states, states[1:]):
			normalC_transmat_[i][j] += 1
			cnt_total += 1
normalC_transmat_ /= cnt_total  # normal class에서의 transitionProbability

# normalC_transmat_을 이용하여, 각 data별, 충격에 관련된 행동이라 의심되는 point marking
thrs = 0.002  # 각 action data에서 transitionProbability가 0.01보다 낮은 (state -> state)가 진행되는 경우, marking
# thrs = 0.001

def abnormalDetect(hiddens):
	for actionNum in action_dic.keys():
		print(action_dic[actionNum], end='\t')
		if actionNum in normalC:
			print(' ')
		else:
			print('Abnormal')

		num_abnormal = 0
		for states in hiddens[actionNum]:
			cnt_overThrs = 0
			for i, j in zip(states, states[1:]):
				if normalC_transmat_[i][j] < thrs:
					cnt_overThrs += 1
			# print(action_dic[actionNum], cnt_overThrs)
			if cnt_overThrs > 0:
				num_abnormal += 1
		try:
			print('Abnormal Detect: {}% ({}/{})'.format(100*num_abnormal/len(hiddens[actionNum]), num_abnormal, len(hiddens[actionNum])))
		except ZeroDivisionError:
			print('Zero #')
		print(' ')

print('===== Train =====')
abnormalDetect(hiddens_tr)

print('=====  Test =====')
abnormalDetect(hiddens_te)

# plt.show()
pdb.set_trace()


## threshold 넘기는 부분 tracking해서, 그래프에 표시만 한번 해보고,
## 괜찮으면 쓰고, 
## 아니면 그냥 성능만 어떻게 파라미터 조절해서 좀 바꾸고, 정리하자.
