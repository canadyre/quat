import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd

import csv


##### AP #######
classes = ["aeroplane", "bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person", "train"]

attacks = ['clean', 'fabrication16', 'fabrication32', 'fabrication8', 'mislabeling16', 'mislabeling32', 'mislabeling8', 'untargeted16', 'untargeted32', 'untargeted8', 'vanishing16', 'vanishing32', 'vanishing8']

attacks_match = ['clean', 'fab16', 'fab32', 'fab8', 'mislabel16', 'mislabel32', 'mislabel8', 'untarget16', 'untarget32', 'untarget8', 'vanish16', 'vanish32', 'vanish8']


################## Full mAP ###############################
map = pd.read_csv('./test.csv')

reg_full = map.loc[(map['model'] == 'regtrain') & (map['tiny/full'] == 'full')]
reg_tiny = map.loc[(map['model'] == 'regtrain') & (map['tiny/full'] == 'tiny')]

map = map.groupby(['tiny/full','model'])

# models_tiny = []#['regtrain_tiny', 'advtrain8_tiny', 'advtrain16_tiny', 'advtrain32_tiny'] # 'regtrain_full', , 'advtrain8_full'
models = []

# print(reg_full)
# print(reg_tiny)

for model, frame in map:

    if (str(model[0]) == 'full') & (str(model[1]) != 'regtrain'):
        empty = []
        #print(frame)
        models.append(str(model[1])+'_'+str(model[0]))
        #print(models_full)
        for a in attacks:
            test = frame.loc[(frame['attack'] == a)]
            #print(test)
            test2 = test['map']
            reg = reg_full.loc[(reg_full['attack'] == a)]
            reg2 = reg['map']
            dev = float(test2) - float(reg2)
            #print(test2)
            empty.append(dev)
        plt.plot(attacks, empty, marker='.')
        # with open('deviation_full.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     for a,d in zip(attacks,empty):
        #         writer.writerow([models_full, a, d])
        # f.close()

    elif (str(model[0]) == 'tiny') & (str(model[1]) != 'regtrain'):
        empty = []
        #print(frame)
        models.append(str(model[1])+'_'+str(model[0]))
        #print(models_tiny)
        for a in attacks_match:
            test = frame.loc[(frame['attack'] == a)]
            #print(test)
            test2 = test['map']
            reg = reg_tiny.loc[(reg_tiny['attack'] == a)]
            reg2 = reg['map']
            dev = float(test2) - float(reg2)
            #print(test2)
            empty.append(dev)

        plt.plot(attacks, empty, marker='.')
        # with open('deviation_tiny.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     for a,d in zip(attacks,empty):
        #         writer.writerow([models_full, a, d])
        # f.close()


plt.legend(models)

plt.tick_params(axis='x', rotation=28)
plt.xlabel('Attack')
plt.ylabel('Change in mAP from regtrain (%)')
plt.ylim(bottom=-15)
plt.tight_layout()
plt.axhline(0, color='gray')
plt.savefig('./plots_combo/map_deviation.png')
plt.clf()
