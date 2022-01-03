import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse
import random

attacks = [8, 16, 32]
att_mods = ['yolo', 'frcnn', 'ssd300']
attack_names = ['clean', 'fabrication', 'mislabeling', 'vanishing', 'untargeted']


voc_dict = { 
    'aeroplane': 0 ,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19}

class VOCData(torch.utils.data.Dataset):
    def __init__(self, root, attack_model='fasterrcnn', attack_name='fabrication',
                bound=8, p=0.5, adv=False, eval=False, theta=0, delta=0, el=0,
                image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                transforms=None):
        self.p = p
        self.adv = adv
        self.root = root
        self.attack_model = attack_model
        self.attack_name = attack_name
        self.bound = bound
        self.theta = theta
        self.delta = delta
        self.el = el
        self.transforms = transforms
        self.image_set = image_sets
        self.eval = eval
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            self._annopath = os.path.join(rootpath, 'Annotations/')
            self._imgpath = os.path.join(rootpath, 'JPEGImages/')
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # Testing data
        if (self.eval):
            if (self.adv):
                bound = self.bound
                att_name = self.attack_name
                att_mod = self.attack_model
                theta = self.theta
                delta = self.delta
                el = self.el
                if att_name == 'fabrication':
                    if os.path.exists(str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'mislabeling':
                    if os.path.exists(str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'untargeted':
                    if os.path.exists(str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'vanishing':
                    if os.path.exists(str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_test_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'poltergeist':
                    img_path = str(self.root)+'VOC_'+str(att_name)+str(theta)+'_'+str(delta)+'_'+str(el)+'/'+str(img_id[1])+'.jpg'
                else:
                    img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
            else:
                img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
        # Training Data
        else:
            if (self.adv):
                bound = random.choice(attacks)
                att_name = random.choice(attack_names)
                att_mod = random.choice(att_mods)
                if att_name == 'fabrication':
                    if os.path.exists(str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'mislabeling':
                    if os.path.exists(str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'untargeted':
                    if os.path.exists(str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                elif att_name == 'vanishing':
                    if os.path.exists(str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'):
                        img_path = str(self.root)+'VOC_train_'+str(att_mod)+'_TOG-'+str(att_name)+str(bound)+'/'+str(img_id[1])+'.jpg'
                    else:
                        img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
                else:
                    img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'
            else:
                img_path = str(img_id[0])+'/JPEGImages/'+str(img_id[1])+'.jpg'

        bbox_xml_path = str(img_id[0])+'/Annotations/'+str(img_id[1])+'.xml'
        img = Image.open(img_path).convert("RGB")

        # Read file, VOC format dataset label is xml format file
        dom = parse(bbox_xml_path)
        # Get Document Element Object
        data = dom.documentElement
        # Get objects
        objects = data.getElementsByTagName('object')
        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            # Get the contents of the label
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # Is label, mark_type_1 or mark_type_2
            name = voc_dict.get(name)
            labels.append(np.int(name))  # Background label is 0, mark_type_1 and mark_type_2 labels are 1 and 2, respectively

            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Since you are training a target detection network, there is no target [masks] = masks in the tutorial
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        #print(img, target)
        if self.transforms is not None:
            # Note that target (including bbox) is also transformed\enhanced here, which is different from transforms from torchvision import
            # Https://github.com/pytorch/vision/tree/master/references/detectionOfTransforms.pyThere are examples of target transformations when RandomHorizontalFlip
            img, target = self.transforms(img, target)
            #target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.ids)
