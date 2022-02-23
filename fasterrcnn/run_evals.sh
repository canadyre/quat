#!/bin/bash


# Code to run TOG attacks

# for back in resnet50 mobilenet_v3-large mobilenet_v3-large-320
# do
#   for att in fabrication vanishing mislabeling untargeted
#   do
#     for att_mod in frcnn ssd300 yolo
#     do
#       for b in 8 16 32
#       do
#               echo "===== ADV -> Model: faster_rcnn; Backbone: $back; Attack: $att; Bound: $b; Attacked_model: $att_mod ====="
#               nohup python test_simple.py --model faster_rcnn --adv True --eval True --backbone ${back} --root_dir /home/canadyre/data/VOCdevkit/ --attack ${att} --attack_bound ${b} --attacked_model ${att_mod} --model_path ./models/fasterrcnn_${back}_epochs25.pth > outputs/clean_train_fasterrcnn_${back}_advimg.out 2>&1 &
#               processID=$!
#               wait $processID
#               echo "===== ADV -> Model: adv_faster_rcnn; Backbone: $back; Attack: $att; Bound: $b; Attacked_model: $att_mod ====="
#               nohup python test_simple.py --model faster_rcnn --adv True --eval True --backbone ${back} --root_dir /home/canadyre/data/VOCdevkit/ --attack ${att} --attack_bound ${b} --attacked_model ${att_mod} --model_path ./models/adv_fasterrcnn_${back}_epochs15.pth > outputs/adv_train_fasterrcnn_${back}_advimg.out 2>&1 &
#               processID=$!
#               wait $processID
#
#       done
#     done
#   done
#
#   echo "===== Model: faster_rcnn; Backbone: $back; Attack: $att; Bound: $b; Attacked_model: $att_mod ====="
#   nohup python test_simple.py --model faster_rcnn --eval True --backbone ${back} --root_dir /home/canadyre/data/VOCdevkit/ --model_path ./models/fasterrcnn_${back}_epochs25.pth > outputs/clean_train_fasterrcnn_${back}_clnimg.out 2>&1 &
#   processID=$!
#   wait $processID
#
#   echo "===== Model: adv_faster_rcnn; Backbone: $back; Attack: $att; Bound: $b; Attacked_model: $att_mod ====="
#   nohup python test_simple.py --model faster_rcnn --eval True --backbone ${back} --root_dir /home/canadyre/data/VOCdevkit/ --model_path ./models/adv_fasterrcnn_${back}_epochs15.pth > outputs/adv_train_fasterrcnn_${back}_clnimg.out 2>&1 &
#   processID=$!
#   wait $processID
# done

# Code to run Poltergeist attacks
#
# for back in resnet50 mobilenet_v3-large mobilenet_v3-large-320
# do
#   for i in 0 2 4 6 8 10
#   do
#     for j in 0 0.02 0.04 0.06 0.08 0.1
#     do
#       for k in 0 10 20 30 40 50
#       do
#         echo "===== Model: faster_rcnn; Backbone: $back; Theta: $i, Delta: $j, L: $k ====="
#         nohup python test_simple.py --model faster_rcnn --adv True --eval True --result_save ./results/polt_test.csv --attack poltergeist --theta ${i} --delta ${j} --el ${k} --backbone ${back} --root_dir /home/canadyre/data/VOCdevkit/ --model_path ./models/fasterrcnn_${back}_epochs25.pth > outputs/clean_train_fasterrcnn_${back}_polt.out 2>&1 &
#         processID=$!
#         wait $processID
#         echo "===== Model: adv_faster_rcnn; Backbone: $back; Theta: $i, Delta: $j, L: $k ====="
#         nohup python test_simple.py --model faster_rcnn --adv True --eval True --result_save ./results/polt_test.csv --attack poltergeist --theta ${i} --delta ${j} --el ${k} --backbone ${back} --root_dir /home/canadyre/data/VOCdevkit/ --model_path ./models/adv_fasterrcnn_${back}_epochs15.pth > outputs/adv_train_fasterrcnn_${back}_polt.out 2>&1 &
#         processID=$!
#         wait $processID
#       done
#     done
#   done
# done
