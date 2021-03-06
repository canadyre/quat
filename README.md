# Quartet Adversarial Training (QUAT)
Repo is for a paper that has been accepted ICAA '22. The repo is under development

## Data Download and Preparation
To start, download the VOCDataset, for ease I have linked the [download page](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) that pjreddie includes for Darknet. You will need to download the train and test set for 2007 and 2012.

### Attack data
The code to attack data is location in the [attack](https://github.com/canadyre/quat/tree/main/attack) directory. To run the attacks, you can either run the script, [run_attacks.sh](https://https://github.com/canadyre/quat/tree/main/fasterrcnn/run_attacks.sh), or directly run one specific attack:

```
python tog_attack.py --weights ${att_mod_pth}  --images voc/train.txt --type ${att} \\
  --model_type ${att_mod} --eps ${eps}
```

## YOLOv3
To train and test YOLOv3 models like we did in the paper, the code and training configuration files are in the [yolov3](https://github.com/canadyre/quat/tree/main/yolov3) directory.

### Training
To train the models, we followed the training steps located at [yolo](https://pjreddie.com/darknet/yolo/). We had to make adjustments to the configuration files for training, but otherwise the install of darknet and training models will be the same.

### Testing on Jetson
```
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \\
  -v /home/canadyre:/home/canadyre nvcr.io/nvidia/l4t-ml:r32.6.1-py3
```

## FasterRCNN
To train and test FasterRCNN models like we did in the paper, the code and necessary configuration files are in the [fasterrcnn](https://https://github.com/canadyre/quat/tree/main/fasterrcnn) directory.

### Training
```
python train_simple.py --model faster_rcnn --backbone ${backbone} --root_dir ${data_directory}
```

### Testing on Jetson
```
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \\
  -v /home/canadyre:/home/canadyre nvcr.io/nvidia/l4t-ml:r32.6.1-py3

# Test on attack data
python test_simple.py --model faster_rcnn --adv True --eval True --backbone ${backbone} \\
  --root_dir ${data_dir} --attack ${att} --attack_bound ${b} --attacked_model ${att_mod} \\
  --model_path ${model_path}

# Test on clean data
python test_simple.py --model faster_rcnn --eval True --backbone ${backbone} \\
  --root_dir ${data_dir} --model_path ${model_path}

```
