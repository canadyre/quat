import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd



##### AP #######
classes = ["aeroplane", "bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person", "train"]

msft = pd.read_csv('./test.csv')


msft = msft.sort_values(['attack', 'model'])

msft = msft.groupby(['attack'])

models = ['regtrain_tiny', 'advtrain8_tiny', 'advtrain16_tiny', 'advtrain32_tiny'] # 'regtrain_full', , 'advtrain8_full'

attacks = ['clean', 'fab16', 'fab32', 'fab8', 'mislabel16', 'mislabel32', 'mislabel8', 'untarget16', 'untarget32', 'untarget8', 'vanish16', 'vanish32', 'vanish8']


for attack, frame in msft:
    cols = list(frame.columns.values)
    x_coor = cols[5:]
    #attacks.append(attack)
    for m in models:
        test = frame.loc[(frame['model']+'_'+frame['tiny/full'] == m)]
        test2 = test[x_coor]
        empty = []
        for cls in classes:
            empty.append(test2[cls])
        if m == 'regtrain_tiny':
            plt.plot(classes, empty, marker='.')
        else:
            plt.plot(classes, empty, marker='*')
    plt.legend(['regtrain_tiny', 'advtrain8_tiny', 'advtrain16_tiny', 'advtrain32_tiny'])

    plt.tick_params(axis='x', rotation=28, labelsize=16)
    plt.xlabel('Class')
    plt.ylabel('AP (%)')
    plt.tight_layout()
    if attack != 'clean':
        plt.ylim(top=10)

    plt.savefig('./plots_tiny/'+attack+'_tiny.png')
    plt.clf()



########################################################TODO: plot mAP somehow ####################################################################################

###### mAP plotting

map = pd.read_csv('./test.csv')


map = map.sort_values(['attack', 'model', 'tiny/full'])

map = map.groupby(['tiny/full','model'])

models = []#['regtrain_tiny', 'advtrain8_tiny', 'advtrain16_tiny', 'advtrain32_tiny'] # 'regtrain_full', , 'advtrain8_full'


for model, frame in map:
    a_group = map.get_group(model)
    if str(model[0]) == 'tiny':
        empty = []
        models.append(str(model[1])+'_'+str(model[0]))
        for a in attacks:
            test = frame.loc[(frame['attack'] == a)]
            test2 = test['map']
            empty.append(float(test2))

        plt.plot(attacks, empty, marker='*')

plt.legend(models)

plt.tick_params(axis='x', rotation=28, labelsize=16)
plt.xlabel('Attack')
plt.ylabel('mAP (%)')
plt.tight_layout()


plt.savefig('./plots_tiny/tiny_attacks_map.png')
plt.clf()
