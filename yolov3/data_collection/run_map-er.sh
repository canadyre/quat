#!/bin/bash


# for dir in regtrain_tiny advtrain_tiny regtrain_full advtrain_full
# do
#   echo "====================================== Model: $dir, Attack: clean, Attacked Model: clean ======================================"
#   #rm ./mAP/input/detection-results/*
#   cp ./test_${dir}/VOC_test_clean/*.txt ./mAP/input/detection-results
#   python mAP/main.py -np
#   mv ./mAP/output ./mAP/output_polt_clean_${dir}
#   rm ./mAP/input/detection-results/*
# done

# for dir in regtrain_tiny advtrain_tiny regtrain_full advtrain_full
# do
#   for i in 0 2 4 6 8 10
#   do
#     for j in 0 0.02 0.04 0.06 0.08 0.1
#     do
#       for k in 0 10 20 30 40 50
#       do
#         echo "====================================== Model: $dir, Attack: poltergeist,  Theta: $i, Delta: $j, L: $k ======================================"
#         #rm ./mAP/input/detection-results/*
#         cp ./test_polt_${dir}/VOC_test_polt_${i}_${j}_${k}/*.txt ./mAP/input/detection-results
#         python mAP/main.py -np
#         mv ./mAP/output ./mAP/output_polt_${i}_${j}_${k}_${dir}
#         rm ./mAP/input/detection-results/*
#       done
#     done
#   done
# done

for dir in regtrain_tiny advtrain_tiny #regtrain_full advtrain_full
do
  echo "====================================== Model: $dir, Attack: clean, Attacked Model: clean ======================================"
  #rm ./mAP/input/detection-results/*
  cp ./new_test_${dir}/VOC_test_clean/*.txt ./mAP/input/detection-results
  python mAP/main.py -np
  mv ./mAP/output ./mAP/output_new_clean_${dir}
  rm ./mAP/input/detection-results/*
done

for dir in regtrain_tiny advtrain_tiny #regtrain_full advtrain_full
do
  for script in untargeted fabrication mislabeling  vanishing
  do
    for model in frcnn ssd300 yolo
    do
      for b in 8 16 32
      do
        echo "====================================== Model: $dir, Attack: $script, Attacked Model: $model ======================================"
        #rm ./mAP/input/detection-results/*
        cp ./new_test_${dir}/VOC_test_${model}_TOG-${script}${b}/*.txt ./mAP/input/detection-results
        python mAP/main.py -np
        mv ./mAP/output ./mAP/output_new_${model}_${script}${b}_${dir}
        rm ./mAP/input/detection-results/*
      done
    done
  done
done

# for dir in regtrain_tiny advtrain_tiny #regtrain_full advtrain_full
# do
#   echo "====================================== Model: $dir, Attack: clean, Attacked Model: clean ======================================"
#   #rm ./mAP/input/detection-results/*
#   cp ./test_${dir}/VOC_test_clean/*.txt ./mAP/input/detection-results
#   python mAP/main.py -np
#   mv ./mAP/output ./mAP/output_clean_${dir}
#   rm ./mAP/input/detection-results/*
# done
#
# for dir in regtrain_tiny advtrain_tiny regtrain_full advtrain_full
# do
#   for script in untargeted fabrication mislabeling  vanishing
#   do
#     for model in frcnn ssd300 yolo
#     do
#       for b in 8 16 32
#       do
#         echo "====================================== Model: $dir, Attack: $script, Attacked Model: $model ======================================"
#         #rm ./mAP/input/detection-results/*
#         cp ./test_${dir}/VOC_test_${model}_TOG-${script}${b}/*.txt ./mAP/input/detection-results
#         python mAP/main.py -np
#         mv ./mAP/output ./mAP/output_${model}_${script}${b}_${dir}
#         rm ./mAP/input/detection-results/*
#       done
#     done
#   done
# done

### Old IC2E

#
# for dir in  regtrain_full advtrain8_full advtrain16_full advtrain32_full # regtrain_tiny advtrain8_tiny advtrain16_tiny advtrain32_tiny
# do
#   for script in fabrication8 fabrication16 fabrication32 mislabeling8 mislabeling16 mislabeling32 untargeted8 untargeted16 untargeted32 vanishing8 vanishing16 vanishing32
#   do
#     echo "====================================== Model: $dir, Attack: $script ======================================"
#     #rm ./mAP/input/detection-results/*
#     cp ./advtest_${dir}/VOC2007_test_TOG-${script}/*.txt ./mAP/input/detection-results
#     python mAP/main.py -np || break
#     mv ./mAP/output ./mAP/output_${script}_${dir}
#     rm ./mAP/input/detection-results/*
#   done
# done

# for dir in  regtrain_full advtrain8_full advtrain16_full advtrain32_full #regtrain_tiny advtrain8_tiny advtrain16_tiny advtrain32_tiny
# do
#   for script in clean
#   do
#     echo "====================================== Model: $dir, Attack: $script ======================================"
#     #rm ./mAP/input/detection-results/*
#     cp ./cleantest_${dir}/*.txt ./mAP/input/detection-results
#     python mAP/main.py -np || break
#     mv ./mAP/output ./mAP/output_${script}_${dir}
#     rm ./mAP/input/detection-results/*
#   done
# done
