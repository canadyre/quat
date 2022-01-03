import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
#import torchvision.transforms as T
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
import TOG
import _utils as det_utils

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# optional command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default='/home/canadyre/data/VOCdevkit/', help='Directory where data is stored')
parser.add_argument("--save_dir", type=str, default='./models', help='Directory where you want to save your model')
parser.add_argument("--model", type=str, default='faster_rcnn', help='Name of the object detection model you want to use ()')
parser.add_argument("--backbone", type=str, default='resnet50', help='Name of the classification model you want to use (resnet50, vgg16, densenet)')
parser.add_argument("--epochs", type=int, default=15, help='Number of training epochs')
parser.add_argument("--batch_size", type=int, default=4, help='')
parser.add_argument("--workers", type=int, default=0, help='')
parser.add_argument("--num_classes", type=int, default=20, help='')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

workers = args.num_workers
batch_size = args.batch_size
num_classes = args.num_classes
dataset_root = args.root_dir
save_dir = args.save_dir

if args.model == 'faster_rcnn':
    if args.backbone == 'resnet50':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif args.backbone == 'mobilenet_v3-large':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif args.backbone == 'mobilenet_v3-large-320':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

### Other models are under development
# elif args.model == 'retinanet':
#     model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
#     # get number of input features for the classifier
#     in_features = model.backbone.out_channels
#     num_anchors = model.anchor_generator.num_anchors_per_location()[0]
#     # replace the pre-trained head with a new one
#     model.head = RetinaNetHead(in_features, num_anchors, num_classes)
#
# elif args.model == 'ssd300':
#     model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
#     # get number of input features for the classifier
#     out_channels = det_utils.retrieve_out_channels(model.backbone, [300,300])
#     num_anchors = model.anchor_generator.num_anchors_per_location()
#     # replace the pre-trained head with a new one
#     model.head = SSDHead(out_channels, num_anchors, num_classes)
#
# elif args.model == 'ssdlite':
#     model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
#     # get number of input features for the classifier
#     out_channels = det_utils.retrieve_out_channels(model.backbone, [320,320])
#     num_anchors = model.anchor_generator.num_anchors_per_location()
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = SSDLiteHead(out_channels, num_anchors, num_classes)


model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.00001)

train_dataset = VOCData(root=dataset_root, adv=args.adv,
                        transforms = get_transform(train=True))

valid_dataset = VOCData(root=dataset_root,
                            image_sets=[('2007', 'test')],
                            transforms = get_transform(train=False))

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          num_workers=workers,
                          shuffle=True,
                          collate_fn=new_utils.collate_fn,
                          pin_memory=True)

valid_loader = DataLoader(valid_dataset,
                          batch_size=1,
                          num_workers=workers,
                          shuffle=False,
                          collate_fn=new_utils.collate_fn,
                          pin_memory=True)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1,
                                               verbose=True)

num_epochs = args.epochs

for epoch in range(num_epochs):
    print('<========Starting Epoch '+ str(epoch)+'=======>')
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, valid_loader, device=device)
    if num_epochs % 5 == 0:
        print('Saving model')
        if args.model == 'faster_rcnn':
            torch.save(model, str(save_dir)+'/fasterrcnn_'+str(args.backbone)+'_epochs'+str(num_epochs)+'.pth')
            model = torch.load(str(save_dir)+'/fasterrcnn_'+str(args.backbone)+'_epochs'+str(num_epochs)+'.pth')
        else:
            torch.save(model, str(save_dir)+'/'+str(args.model)+'_epochs'+str(num_epochs)+'.pth')
            model = torch.load(str(save_dir)+'/'+str(args.model)+'_epochs'+str(num_epochs)+'.pth')
print("That's it!")
