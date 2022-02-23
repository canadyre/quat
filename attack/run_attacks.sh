#!/bin/bash

# for at in mislabel fabrication untargeted vanishing
# do
#     for eps in 8 16 32
#     do
#         echo "====================================== Train -> Model_type: frcnn, Attack: $at, Epsilon:  ======================================"
#         python tog_attack.py --weights ./weights/frcnn.pth  --images voc/train.txt --type ${at} --model_type frcnn --eps ${eps}
#     done
# done

# for model in ssd300 yolo
# do
#     for at in mislabel fabrication untargeted vanishing
#     do
#         for eps in 8 16 32
#         do
#             echo "====================================== Train -> Model_type: $model, Attack: $at, Epsilon: $eps ======================================"
#             python tog_attack.py --weights ./weights/${model}.h5  --images voc/train.txt --type ${at} --model_type ${model} --eps ${eps}
#         done
#     done
# done

# for at in mislabel fabrication untargeted vanishing
# do
#     for eps in 8 16 32
#     do
#         echo "====================================== Test -> Model_type: frcnn, Attack: $at, Epsilon:  ======================================"
#         python tog_attack.py --weights ./weights/frcnn.pth --image_type test  --images voc/2007_test.txt --type ${at} --model_type frcnn --eps ${eps}
#     done
# done

# for model in yolo ssd300
# do
#     for at in mislabel fabrication untargeted vanishing
#     do
#         for eps in 8 16 32
#         do
#             echo "====================================== Test -> Model_type: $model, Attack: $at, Epsilon: $eps ======================================"
#             python tog_attack.py --weights ./weights/${model}.h5 --image_type test --images voc/2007_test.txt --type ${at} --model_type ${model} --eps ${eps}
#         done
#     done
# done
