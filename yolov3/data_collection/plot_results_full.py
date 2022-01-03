import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd



##### AP #######
classes = ["aeroplane", "bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person", "train"]

msft = pd.read_csv('./test.csv')

msft = msft.sort_values(['attack', 'model'])

attacks = ['clean', 'fabrication16', 'fabrication32', 'fabrication8', 'mislabeling16', 'mislabeling32', 'mislabeling8', 'untargeted16', 'untargeted32', 'untargeted8', 'vanishing16', 'vanishing32', 'vanishing8']

attacks_match = ['clean', 'fab16', 'fab32', 'fab8', 'mislabel16', 'mislabel32', 'mislabel8', 'untarget16', 'untarget32', 'untarget8', 'vanish16', 'vanish32', 'vanish8']

msft = msft.groupby(['attack', 'tiny/full'])

models = ['regtrain_full', 'advtrain8_full', 'advtrain16_full', 'advtrain32_full']

for attack, frame in msft:
    cols = list(frame.columns.values)
    x_coor = cols[5:]

    #attacks.append(attack[0])
    if attack[1] == 'full':
        for m in models:

            test = frame.loc[(frame['model']+'_'+frame['tiny/full'] == m)]

            test2 = test[x_coor]

            empty = []
            for cls in classes:
                empty.append(test2[cls])

            plt.plot(classes, empty, marker='.')

        plt.legend(['regtrain_full', 'advtrain8_full', 'advtrain16_full', 'advtrain32_full'])

        plt.tick_params(axis='x', rotation=28)
        plt.xlabel('Class')
        plt.ylabel('AP (%)')
        plt.tight_layout()
        if attack[0] != 'clean':
            plt.ylim(top=30)

        plt.savefig('./plots_full/'+attack[0]+'_full.png')
        plt.clf()



################## Full mAP ###############################
map = pd.read_csv('./test.csv')

map = map.sort_values(['model', 'tiny/full'])

map = map.groupby(['tiny/full','model'])

models = []#['regtrain_tiny', 'advtrain8_tiny', 'advtrain16_tiny', 'advtrain32_tiny'] # 'regtrain_full', , 'advtrain8_full'

for model, frame in map:
    a_group = map.get_group(model)
    if str(model[0]) == 'full':
        empty = []
        #print(frame)
        models.append(str(model[1])+'_'+str(model[0]))
        #print(attacks)
        for a in attacks:
            test = frame.loc[(frame['attack'] == a)]
            #print(test)
            test2 = test['map']
            #print(test2)
            empty.append(float(test2))

        plt.plot(attacks_match, empty, marker='*')

plt.legend(models)

plt.tick_params(axis='x', rotation=28)
plt.xlabel('Attack')
plt.ylabel('mAP (%)')
plt.tight_layout()


plt.savefig('./plots_full/full_attacks_map.png')
plt.clf()
