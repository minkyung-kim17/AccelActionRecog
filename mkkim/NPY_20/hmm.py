import pdb
import os
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import argparse

action_dic = {
              '5': 'Throw',
			  '7': 'Basketball shoot',
			  '12': 'Bowling',
			  '15': 'Tennis swing',
			  '16': 'Arm curl',
			  '18': 'Push',
			  # '21': 'Pickup and throw',  # pickup은 normal이랑 겹치고, throw는 throw랑 겹침
              '100': 'Normal',
              '200': 'Abnormal'}

windowSize = 20  # 1s
windowInterval = 4
numStates = 12

trte_rate = 0.95
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
data_tr = {}
models = {}
for actionNum in action_dic.keys():
    actionFiles = [x for x in AllFiles_tr if x.split('_')[0][1:] == actionNum]

    actionDataWinAll = []
    for file in actionFiles:
        data = np.load(os.path.join('./', file))

        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]

        a_tot = np.sqrt(ax**2 + ay**2 + az**2)

        actionDataWin = []
        for i in range(0, len(a_tot), windowInterval):
            if i+windowSize <= len(a_tot):
                actionDataWin.append(a_tot[i:i+windowSize])
        actionDataWinAll.append(np.array(actionDataWin))
    actionDataWinAll = np.array(actionDataWinAll)

    actionDataWinAllMS = []  # convert to (mean, std)
    for i in range(len(actionDataWinAll)):
        actionDataWin = actionDataWinAll[i]
        actionDataWinAllMS.append(np.vstack((np.mean(actionDataWin, axis=1), np.std(actionDataWin, axis=1))).T)
    actionDataWinAllMS = np.array(actionDataWinAllMS)

    data_tr[actionNum] = actionDataWinAllMS

## HMM modeling
for actionNum in action_dic.keys():
    data = np.random.random((0, 2))  # (mean, std)
    datalen = []
    for i in range(len(data_tr[actionNum])):
        dataWin = data_tr[actionNum][i]
        datalen.append(len(dataWin))
        data = np.concatenate([data, dataWin])

    models[actionNum] = hmm.GaussianHMM(n_components=numStates).fit(data, datalen)

## TR data check  --> training은 각 action별 hmm 모델에서 평균 score가 제일 높게 나옴.
scoreAll = {}
for actionNum in action_dic.keys():
    # print(actionNum)

    scores = []
    for i in range(len(data_tr[actionNum])):
        dataWin = data_tr[actionNum][i]

        temp = []  # action data 하나에 대해서
        for aNum, model in models.items():
            temp.append([int(aNum), float(model.score(dataWin))])

        temp = np.array(sorted(temp, key=lambda x: x[0]))
        scores.append(temp[:, 1])  # score만, 순서는 어차피 numAction이 작은 순서대로임.
    scoreAll[actionNum] = np.array(scores)

# print('Training data model score check (mean)')
print('Training')
print('  \t 5\t\t\t 7\t\t\t12\t\t\t15\t\t\t16\t\t\t18\t\t\t21')
for actionNum in sorted(map(int, action_dic.keys())):
    scores = scoreAll[str(actionNum)]

    print('{}\t'.format(actionNum), end='')
    for score in np.mean(scores, axis=0):
        print('{}\t'.format(score), end='')
    print(' ')

## TE data check
print('\nTest')
acc1 = 0
acc2 = 0
acc3 = 0
for file in AllFiles_te:
    data = np.load(os.path.join('./', file))

    ax = data[:, 0]
    ay = data[:, 1]
    az = data[:, 2]

    a_tot = np.sqrt(ax**2 + ay**2 + az**2)

    actionDataWin = []
    for i in range(0, len(a_tot), windowInterval):
        if i+windowSize <= len(a_tot):
            actionDataWin.append(a_tot[i:i+windowSize])
    actionDataWin = np.array(actionDataWin)
    actionDataWinMS = np.vstack((np.mean(actionDataWin, axis=1), np.std(actionDataWin, axis=1))).T

    scores_te = np.array([[int(aNum), float(model.score(actionDataWinMS))] for aNum, model in models.items()])
    recogs = sorted(scores_te, key=lambda x: x[1], reverse=True)

    target = int(file.split('_')[0][1:])
    recog = int(recogs[0][0])
    recog2 = int(recogs[1][0])
    recog3 = int(recogs[2][0])

    print('TARGET CLASS {:>2} \t RECOG CLASS {:>4} {:>4} {:>4}'.format(target, recog, recog2, recog3))

    if target == recog3:
        acc3 += 1
    elif target == recog2:
        acc2 += 1
    elif target == recog:
        acc1 += 1

print('\nTop 1 ACC: {}'.format(acc1/len(AllFiles_te)))
print('Top 2 ACC: {}'.format((acc1+acc2)/len(AllFiles_te)))
print('Top 3 ACC: {}'.format((acc1+acc2+acc3)/len(AllFiles_te)))
