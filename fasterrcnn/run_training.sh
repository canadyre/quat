#!/bin/bash

for back in resnet50 mobilenet_v3-large mobilenet_v3-large-320
do
        #echo "===== Model: faster_rcnn; Backbone: $back ====="
        #nohup python train_simple_visdrone.py --model faster_rcnn --adv True --backbone ${back} --root_dir /home/ubuntu/robert_c/VOCdevkit/ > outputs/new_adv_train_fasterrcnn_${back}.out 2>&1 & 
        #processID=$!
        #wait $processID

        echo "===== Model: faster_rcnn; Backbone: $back ====="
        nohup python train_simple_visdrone.py --model faster_rcnn --backbone ${back} --root_dir /home/canadyre/VisDrone/ > outputs/visdrone_clean_train_fasterrcnn_${back}.out 2>&1 & 
        processID=$!
        wait $processID
done
