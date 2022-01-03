import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


yolo_results = '/home/canadyre/main_Dropbox/IC2E_code/mAP/test.csv'
yolo_results_time = '/home/canadyre/main_Dropbox/IC2E_code/time_per_image.csv'

col_names = ['obj_mod', 'back', 'clean_adv_train', 'attack', 'attack_bound', 'attack_model', 'mAP', 'mAP50','0','1','2','3','4','5','6','7','8','9','pre-process', 'eval_time']

comp_yolo_array = []

# 'back', 'clean_adv_train', 'attack', 'attack_bound', 'attack_model', 'eval_time'
time_array = []
with open(yolo_results_time, newline='') as csvfile0:
    yolo_time = list(csv.reader(csvfile0))
    for line in yolo_time:
        if line[0][0:3] == 'reg':
            temp_train = 'clean'
        else:
            temp_train = 'adv'
        if line[4] == 'clean':
            temp = [line[1], temp_train, line[4], line[4], line[4], line[5]]
        else:
            temp = [line[1], temp_train, line[5], line[6], line[4], line[7]]
        time_array.append(temp)

#print(time_array)
with open(yolo_results, newline='') as csvfile:
    yolo_data = list(csv.reader(csvfile))
    for line in yolo_data:
        #print(str(line[3][0])+str(line[3][2])+str(line[3][2]))

        if (line[3]) == 'regtrain':
            temp_train = 'clean'
        else:
            temp_train = 'adv'

        temp_att_mod = str(line[1])

        #print(line[2])
        if str(line[2][-1]) == '8':
            temp_bound = 8
            temp_attack = line[2][0:-1]
        elif (str(line[2][-2])+str(line[2][-1])) == '16':
            temp_bound = 16
            temp_attack = line[2][0:-2]
        elif (str(line[2][-2])+str(line[2][-1])) == '32':
            temp_bound = 32
            temp_attack = line[2][0:-2]
        else:
            temp_bound = 0
            temp_attack = 'clean'
            temp_att_mod = 'clean'
        temp_back = str(line[4])

        temp_time = 0
        for line2 in time_array:
            # print((temp_back, line2[0]))
            # print((temp_train, line2[1]))
            # print((temp_bound, line2[3]))
            # print((temp_att_mod, line2[4]))
            if (line2[3] == 'clean'):
                b = 0
            else:
                b = int(line2[3])
            if (temp_back == line2[0]) and (temp_train == line2[1]) and (temp_bound == b) and (temp_att_mod == line2[4]):
                #print('here')
                temp_time = line2[5]
                break
        #print(str(float(line[5])/100))
        temp = ['yolov3', temp_back, temp_train, temp_attack, temp_bound, temp_att_mod, '', str(float(line[5])/100), '','','','','','','','','','','', temp_time]
        comp_yolo_array.append(temp)



frcnn_results = '/home/canadyre/adv_fusion/results/test.csv'
with open(frcnn_results, newline='') as csvfile2:
    frcnn_data = list(csv.reader(csvfile2))
    for line in frcnn_data:
        temp = line[2].split('_')
        #print(temp)
        if temp[0] == 'adv':
            line[2] = 'adv'
        else:
            line[2] = 'clean'
        comp_yolo_array.append(line)

#print(comp_yolo_array)

pd.set_option('display.max_rows', None)
df = pd.DataFrame(data=comp_yolo_array, columns=col_names)
#print(df)
df['model'] = df[['obj_mod', 'back', 'clean_adv_train']].agg('-'.join, axis=1)
#df[['attack_bound', 'mAP', 'mAP50','0','1','2','3','4','5','6','7','8','9','pre-process', 'eval_time']] = df[['attack_bound', 'mAP', 'mAP50','0','1','2','3','4','5','6','7','8','9','pre-process', 'eval_time']].apply(pd.to_numeric)
df = df.drop(columns=['obj_mod', 'back', 'clean_adv_train', '0','1','2','3','4','5','6','7','8','9'])
df[['attack_bound', 'mAP', 'mAP50', 'pre-process', 'eval_time']] = df[['attack_bound', 'mAP', 'mAP50', 'pre-process', 'eval_time']].apply(pd.to_numeric)


mods = df['model'].unique()
#print(mods)
attack_mods = ['frcnn'] # , 'ssd300', 'yolo'
attac = ['fabrication', 'mislabeling', 'untargeted', 'vanishing']
bnd = [8,16,32]

df = df.groupby(['model', 'attack_model', 'attack', 'attack_bound'])



for m in mods:
    map = []
    att_array = []
    if ((m == 'yolo5l-n-clean') | (m == 'yolo5l-n-adv') | (m == 'yolo5s-n-clean') | (m == 'yolo5s-n-adv')):
        continue
    #print(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    map.append(df.first().loc[(m,'clean','clean', 0)]['mAP50'])
    att_array.append('clean')
    # for a in attack_mods:
    for at in attac:
        for bn in bnd:
            #print(df.first().loc[(m,'frcnn',at,bn)]['mAP'])
            map.append(df.first().loc[(m,'frcnn',at,bn)]['mAP50'])
            att_array.append(str(at)+'_'+str(bn))
    plt.plot(att_array, map, label = m)
plt.legend(fontsize=7, ncol=2)
plt.xticks(rotation = 65)
plt.ylim(top=1.0, bottom=0.0)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP')
plt.title('FasterRCNN')
plt.tight_layout()
plt.savefig('./plots/frcnn_attacks.png')
#plt.show()
plt.clf()

for m in mods:
    map = []
    att_array = []
    if ((m == 'yolo5l-n-clean') | (m == 'yolo5l-n-adv') | (m == 'yolo5s-n-clean') | (m == 'yolo5s-n-adv')):
        continue
    #print(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    map.append(df.first().loc[(m,'clean','clean', 0)]['mAP50'])
    att_array.append('clean')
    # for a in attack_mods:
    for at in attac:
        for bn in bnd:
            #print(df.first().loc[(m,'ssd300',at,bn)]['mAP'])
            map.append(df.first().loc[(m,'ssd300',at,bn)]['mAP50'])
            att_array.append(str(at)+'_'+str(bn))
    plt.plot(att_array, map, label = m)
plt.legend(fontsize=7, ncol=2)
plt.xticks(rotation = 65)
plt.ylim(top=1.0, bottom=0.0)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP')
plt.title('SSD300')
plt.tight_layout()
plt.savefig('./plots/ssd300_attacks.png')
#plt.show()

plt.clf()

for m in mods:
    map = []
    att_array = []
    if ((m == 'yolo5l-n-clean') | (m == 'yolo5l-n-adv') | (m == 'yolo5s-n-clean') | (m == 'yolo5s-n-adv')):
        continue
    #print(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    map.append(df.first().loc[(m,'clean','clean', 0)]['mAP50'])
    att_array.append('clean')
    # for a in attack_mods:
    for at in attac:
        for bn in bnd:
            #print(df.first().loc[(m,'yolo',at,bn)]['mAP'])
            map.append(df.first().loc[(m,'yolo',at,bn)]['mAP50'])
            att_array.append(str(at)+'_'+str(bn))
    plt.plot(att_array, map, label = m)
plt.legend(fontsize=7, ncol=2)
plt.xticks(rotation = 65)
plt.ylim(top=1.0, bottom=0.0)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP')
plt.title('YOLO')
plt.tight_layout()
plt.savefig('./plots/yolo_attacks.png')
#plt.show()
