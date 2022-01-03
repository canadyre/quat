from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util_tiny import *
import argparse
import os
import os.path as osp
from darknet_tiny import Darknet
import pickle as pkl
import pandas as pd
import random
import csv

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import new_utils as utils


def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3-tiny.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--clean", dest = 'clean', help =
                        "clean, adv",
                        default = "adv", type = str)

    return parser.parse_args()

args = arg_parse()
images = args.images
clean = args.clean
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 20
classes = load_classes("data/voc.names")

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()

read_dir = time.time()
#Detection phase
if (clean == 'clean'):
    f = open(images, "r")
    im_list = []
    for line in f:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        im_list.append(line_list)
    f.close()
else:
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
if (clean == 'clean'):
    imlist = []
    for x in im_list:
        imlist.append(x[0])
    loaded_ims = [cv2.imread(x) for x in imlist]
else:
    loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]

write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()
time_array = []
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
#load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
            time_array.append((end - start)/batch_size)
            print((end - start)/batch_size)
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist

    if not write:                      #If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")
        time_array.append((end - start)/batch_size)
        print((end - start)/batch_size)
    if CUDA:
        torch.cuda.synchronize()
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("cfg/pallete", "rb"))

draw = time.time()

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX, 1, [225,255,255], 1);
    return img

det_names = pd.Series(imlist).apply(lambda x: "{}/{}".format(args.det,x.split("/")[-1]))

end = time.time()
## Detection-results txt files
det_list = []
for name in det_names:
    file = name.split('.')
    file = file[0]+'.txt'
    det_list.append(file)

for o in output:
    print(o)
    img_num = int(o[0])
    img_file = det_names[img_num]
    clss = int(o[-1])
    label = classes[clss]
    conf = (o[5].cpu()).numpy()
    c1 = (o[1].cpu()).numpy()
    c2 = (o[2].cpu()).numpy()
    c3 = (o[3].cpu()).numpy()
    c4 = (o[4].cpu()).numpy()

    with open(det_list[img_num], 'a') as file1:
        file1.write(str(label)+' '+str(conf)+' '+str(c1)+' '+str(c2)+' '+str(c3)+' '+str(c4)+'\n')
        file1.close()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

time_per_image = (end - load_batch)/len(imlist)

# Create empty text files for non detections
for f in imlist:
    temp = f.split('/')[-1]
    temp2 = os.path.join(args.det, temp.replace('.jpg', '.txt'))
    file_exists = os.path.isfile(temp2)
    if not file_exists:
        open(temp2, 'w+').close()
    else:
        continue

list = [str(args.weightsfile), str(args.det), sum(time_array)/len(time_array)]
with open('time_per_image_tinyyoloredo.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(list)

torch.cuda.empty_cache()
