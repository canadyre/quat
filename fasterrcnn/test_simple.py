import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.datasets as datasets
from torchvision.models import detection
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from engine import train_one_epoch, evaluate
import new_utils
from voc_class import VOCData
import argparse
import os
import csv
import TOG
import _utils as det_utils
import transforms as T

# optional command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default='/home/canadyre/data/VOCdevkit/', help='Directory where data is stored')
parser.add_argument("--save_dir", type=str, default='./models', help='Directory where you want to save your model')
parser.add_argument("--result_save", type=str, default='./results/new_test.csv', help='Directory where you want to save your model')
parser.add_argument("--model", type=str, default='faster_rcnn', help='Name of the object detection model you want to use ()')
parser.add_argument("--model_path", type=str, default='faster_rcnn', help='Name of the object detection model you want to use ()')
parser.add_argument("--attack", type=str, default='fabrication', help='Name of the object detection model you want to use ()')
parser.add_argument("--attack_bound", type=int, default=8, help='Name of the object detection model you want to use ()')
parser.add_argument("--attacked_model", type=str, default='faster_rcnn', help='Name of the object detection model you want to use ()')
parser.add_argument("--backbone", type=str, default='resnet50', help='Name of the classification model you want to use (resnet50, vgg16, densenet)')
parser.add_argument("--adv", type=bool, default=False, help='adversarial training')
parser.add_argument("--eval", type=bool, default=True, help='eval or train')
parser.add_argument("--epochs", type=int, default=10, help='')
parser.add_argument("--batch_size", type=int, default=4, help='')
parser.add_argument("--theta", dest = "theta", help = "", default = 5)
parser.add_argument("--delta", dest = "delta", help = "", default = 0.05)
parser.add_argument("--el", dest = "el", help = "", default = 25)
parser.add_argument("--workers", type=int, default=0, help='')
parser.add_argument("--num_classes", type=int, default=20, help='')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

workers = args.workers
batch_size = args.batch_size
num_classes = args.num_classes
dataset_root = args.root_dir

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if args.model == 'faster_rcnn':
    if args.backbone == 'resnet50':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    elif args.backbone == 'mobilenet_v3-large':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    elif args.backbone == 'mobilenet_v3-large-320':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = torch.load(args.model_path)


model.to(device)

valid_dataset = VOCData(root=dataset_root, attack_model=args.attacked_model,
                            attack_name=args.attack, bound=args.attack_bound,
                            eval=True, adv=args.adv, theta=args.theta, delta=args.delta, el=args.el,
                            image_sets=[('2007', 'test_subset')],
                            transforms = get_transform(train=False))

valid_loader = DataLoader(valid_dataset,
                          batch_size=1,
                          num_workers=workers,
                          shuffle=False,
                          collate_fn=new_utils.collate_fn,
                          pin_memory=True)


test, mod_time, eval_time = evaluate(model, valid_loader, art=False, device=device)
data_array = []
data_array.append(args.model)
data_array.append(args.backbone)
data_array.append(args.model_path)

if (args.adv):
    if (args.attack == 'poltergeist'):
        data_array.append(args.attack)
        data_array.append(str(args.delta)+'_'+str(args.theta))
        data_array.append(args.el)
    else:
        data_array.append(args.attack)
        data_array.append(args.attack_bound)
        data_array.append(args.attacked_model)

elif (args.adv == False):
    data_array.append('clean')
    data_array.append(0)
    data_array.append('clean')

for val in test.stats:
        data_array.append(val)

data_array.append(mod_time)
data_array.append(eval_time)

data_array = [data_array]

if os.path.exists(args.result_save):
    file = open(args.result_save, 'a+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data_array)
else:
    file = open(args.result_save, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data_array)


print("That's it!")
