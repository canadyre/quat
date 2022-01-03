import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

yolo_base = '/home/canadyre/yolov5/runs/val/'
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

pd.set_option('display.max_rows', None)
df = pd.DataFrame(data=comp_yolo_array, columns=col_names)
df['model'] = df[['obj_mod', 'back', 'clean_adv_train']].agg('-'.join, axis=1)
#df[['attack_bound', 'mAP', 'mAP50','0','1','2','3','4','5','6','7','8','9','pre-process', 'eval_time']] = df[['attack_bound', 'mAP', 'mAP50','0','1','2','3','4','5','6','7','8','9','pre-process', 'eval_time']].apply(pd.to_numeric)
df = df.drop(columns=['obj_mod', 'back', 'clean_adv_train', '0','1','2','3','4','5','6','7','8','9'])
df[['attack_bound', 'mAP', 'mAP50', 'pre-process', 'eval_time']] = df[['attack_bound', 'mAP', 'mAP50', 'pre-process', 'eval_time']].apply(pd.to_numeric)


mods = df['model'].unique()
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
plt.legend()
plt.xticks(rotation = 65)
plt.ylim(top=1.0)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP50')
plt.title('FasterRCNN')
plt.tight_layout()
plt.savefig('./plots/frcnn_attacks_map50.png')
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
plt.legend()
plt.xticks(rotation = 65)
plt.ylim(top=1.0)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP50')
plt.title('SSD300')
plt.tight_layout()
plt.savefig('./plots/ssd300_attacks_map50.png')
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
plt.legend()
plt.xticks(rotation = 65)
plt.ylim(top=1.0)
plt.xlabel('Attack Type and Bound')
plt.ylabel('mAP50')
plt.title('YOLO')
plt.tight_layout()
plt.savefig('./plots/yolo_attacks_map50.png')
#plt.show()

#new_df = df.sort_values(['obj_mod', 'back', 'clean_adv_train', 'attack', 'attack_bound', 'attack_model']) #'obj_mod', 'back', 'model_name',
#print(df.first().loc[mods[4]]['mAP'])
#print(df.size())




# models = 'faster_rcnn'
# backbone = ['mobilenet_v3-large', 'mobilenet_v3-large-320', 'resnet50']
# train_type = ['adv', 'clean']
# attacks = ['fabrication', 'mislabeling', 'untargeted', 'vanishing']
# bounds = [8,16,32]
# map_dict = {'model': []}
# map50_dict = {'model': []}
# eval_dict = {'model': []}
#
# for back in backbone:
#     for type in train_type:
#         temp_clean = new_df.loc[(new_df['obj_mod'] == models) & (new_df['back'] == back)
#                 & (new_df['clean_adv_train'] == type) & (new_df['attack'] == 'clean')]
#         #print(temp_clean)
#
#         map_dict['model'].append(str(models)+str(back)+str(type))
#         map50_dict['model'].append(str(models)+str(back)+str(type))
#         eval_dict['model'].append(str(models)+str(back)+str(type))
#
#         print(temp_clean['mAP'].astype('float32'))
#         map_avg = temp_clean['mAP'].astype('float32')
#         map50_avg = temp_clean['mAP50'].astype('float32')
#         avg_eval = temp_clean['eval_time'].astype('float32')
#
#         map_dict['clean'] = map_avg
#
#         map50_dict['clean'] = map50_avg
#
#         eval_dict['clean'] = avg_eval
#
#         for att in attacks:
#             for b in bounds:
#                 #print(back, type, att, b)
#                 temp = new_df.loc[(new_df['obj_mod'] == models) & (new_df['back'] == back)
#                         & (new_df['clean_adv_train'] == type) & (new_df['attack'] == att)
#                         & (new_df['attack_bound'] == b)]
#                 #print(temp)
#                 #print(temp['mAP'].dtype(int32))
#                 map_avg = temp['mAP'].astype('float32').mean()
#                 map50_avg = temp['mAP50'].astype('float32').mean()
#                 avg_eval = temp['eval_time'].astype('float32').mean()
#
#                 map_dict[str(att)+str(b)] = map_avg
#
#                 map50_dict[str(att)+str(b)] = map50_avg
#
#                 eval_dict[str(att)+str(b)] = avg_eval
#
# print(map_dict)
# map_df = pd.DataFrame.from_dict(map_dict)
# map50_df = pd.DataFrame.from_dict(map50_dict)
# eval_df = pd.DataFrame.from_dict(eval_dict)
#
# print(map_df)
