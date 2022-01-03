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

dir_list = next(os.walk('./output_transfer/'))[1]

## TODO: split dir_list to get experiment info
## TODO: grab mAP and AP for each class and save to one csv file
## TODO (if time): grab other data like false positive/false negative, etc.
#print(dir_list)

classes = ["type", "attack", "model", "tiny/full", "map", "aeroplane", "bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person", "train"]

with open('test_transfer.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(classes)

    for dir in dir_list:

        with open('/home/canadyre/Dropbox/IC2E_code/mAP/output_transfer/'+dir+'/output.txt','r') as myfile:
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
        for names in temp_dir:
            new_list.append(names)

        for element in lines:
            temp = element.split(' ')
            if temp[0] == 'mAP':
                new_list.append(temp[-1].rstrip('%'))

            else:
                new_list.append(temp[0].rstrip('%'))


        writer.writerow(new_list)
