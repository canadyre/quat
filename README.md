# Quartet Adversarial Training (QUAT)
Repo is for a paper that has been accepted ICAA '22. The repo is under development

## Data Download and Preparation
To start, download the VOCDataset, for ease I have linked the [download page](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) that pjreddie includes for Darknet. You will need to download the train and test set for 2007 and 2012.

### Attack data
The code to attack data is location in the [attack](https://github.com/canadyre/quat/attack) directory.

## YOLOv3
To train and test YOLOv3 models like we did in the paper, the code and necessary configuration files are in the [yolov3](https://github.com/canadyre/quat/yolov3) directory.

### Training
To train the models, we followed the training steps located at [yolo](https://pjreddie.com/darknet/yolo/). We had to make adjustments to the configuration files for training, but otherwise the install of darknet and training models will be the same.

## FasterRCNN
To train and test FasterRCNN models like we did in the paper, the code and necessary configuration files are in the [fasterrcnn](https://github.com/canadyre/quat/fasterrcnn) directory.

### Training
