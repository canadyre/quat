import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

yolo_base = '/home/canadyre/jetson_data/val/'
yolo_results = os.listdir(yolo_base)

col_names = ['obj_mod', 'back', 'clean_adv_train', 'attack', 'attack_bound', 'attack_model', 'mAP', 'mAP50','0','1','2','3','4','5','6','7','8','9','pre-process', 'eval_time']

comp_yolo_array = []
for att in yolo_results:
    yolo_array = []
    res_file = yolo_base+att+'/results.csv'
    with open(res_file, newline='') as csvfile:

        data = list(csv.reader(csvfile))

        temp = att
        info = temp.split('_')
        if len(info) == 4:
            yolo_array.append(info[0])
            yolo_array.append('n')
            yolo_array.append(info[2])
            yolo_array.append(info[3])
            yolo_array.append('0')
            yolo_array.append(info[3])
        else:
            yolo_array.append(info[0])
            yolo_array.append('n')
            yolo_array.append(info[2])
            yolo_array.append(info[3])
            yolo_array.append(info[5])
            yolo_array.append(info[4])

        for val in data:
            yolo_array.append(val[3])
            yolo_array.append(val[2])
            for x in range(10):
                yolo_array.append(x)
            yolo_array.append(val[4])
            yolo_array.append(val[5])

    comp_yolo_array.append(yolo_array)

#
#print(comp_yolo_array[0])

frcnn_results = '/home/canadyre/jetson_data/test.csv'
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

pd.set_option('display.max_rows', None)
df = pd.DataFrame(data=comp_yolo_array, columns=col_names)
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
    map.append(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    att_array.append('clean')
    # for a in attack_mods:
    for at in attac:
        for bn in bnd:
            #print(df.first().loc[(m,'frcnn',at,bn)]['mAP'])
            map.append(df.first().loc[(m,'frcnn',at,bn)]['mAP'])
            att_array.append(str(at)+'_'+str(bn))
    plt.plot(att_array, map, label = m)
plt.legend()
plt.xticks(rotation = 65)
plt.ylim(top=0.5)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP')
plt.title('FasterRCNN')
plt.tight_layout()
plt.savefig('./plots/frcnn_attacks_jetson.png')
#plt.show()
plt.clf()

for m in mods:
    map = []
    att_array = []
    if ((m == 'yolo5l-n-clean') | (m == 'yolo5l-n-adv') | (m == 'yolo5s-n-clean') | (m == 'yolo5s-n-adv')):
        continue
    #print(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    map.append(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    att_array.append('clean')
    # for a in attack_mods:
    for at in attac:
        for bn in bnd:
            #print(df.first().loc[(m,'ssd300',at,bn)]['mAP'])
            map.append(df.first().loc[(m,'ssd300',at,bn)]['mAP'])
            att_array.append(str(at)+'_'+str(bn))
    plt.plot(att_array, map, label = m)
plt.legend()
plt.xticks(rotation = 65)
plt.ylim(top=0.5)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP')
plt.title('SSD300')
plt.tight_layout()
plt.savefig('./plots/ssd300_attacks_jetson.png')
#plt.show()

plt.clf()

for m in mods:
    map = []
    att_array = []
    if ((m == 'yolo5l-n-clean') | (m == 'yolo5l-n-adv') | (m == 'yolo5s-n-clean') | (m == 'yolo5s-n-adv')):
        continue
    #print(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    map.append(df.first().loc[(m,'clean','clean', 0)]['mAP'])
    att_array.append('clean')
    # for a in attack_mods:
    for at in attac:
        for bn in bnd:
            #print(df.first().loc[(m,'yolo',at,bn)]['mAP'])
            map.append(df.first().loc[(m,'yolo',at,bn)]['mAP'])
            att_array.append(str(at)+'_'+str(bn))
    plt.plot(att_array, map, label = m)
plt.legend()
plt.xticks(rotation = 65)
plt.ylim(top=0.5)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP')
plt.title('YOLO')
plt.tight_layout()
plt.savefig('./plots/yolo_attacks_jetson.png')
#plt.show()
