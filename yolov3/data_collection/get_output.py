import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import re
import csv

import numpy as np

dir_list = next(os.walk('.'))[1]
dir_list.remove('input')
dir_list.remove('old_outputs')
dir_list.remove('old_plts')
dir_list.remove('old_results')
dir_list.remove('scripts')
dir_list.remove('.git')
dir_list.remove('TOG_outputs')
dir_list.remove('outputs_polt')
dir_list.remove('old_jetson_outputs')


## TODO: split dir_list to get experiment info
## TODO: grab mAP and AP for each class and save to one csv file
## TODO (if time): grab other data like false positive/false negative, etc.


classes = ["type", "vict_mod", "attack", "model", "tiny/full", "map", "aeroplane", "bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person", "train"]

with open('new_tiny_test.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(classes)

    for dir in dir_list:

        with open('./'+dir+'/output.txt','r') as myfile:
            lines = []
            for myline in myfile:
                map_pat = re.compile("mAP =")
                if map_pat.search(myline) != None:
                    lines.append(myline.rstrip('\n'))
                for cls in classes:
                    pattern = re.compile(cls+" AP")
                    if pattern.search(myline) != None:
                        lines.append(myline.rstrip('\n'))
        new_list = []
        temp_dir = dir.split('_')
        #print(len(temp_dir))
        count = 0
        for names in temp_dir:
            if len(temp_dir) == 4:
                if count == 1:
                    new_list.append(names)
            new_list.append(names)
            count += 1

        for element in lines:
            temp = element.split(' ')
            if temp[0] == 'mAP':
                new_list.append(temp[-1].rstrip('%'))

            else:
                new_list.append(temp[0].rstrip('%'))


        writer.writerow(new_list)
