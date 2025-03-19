#!/bin/bash

GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='disjoint'
TASKNAME='15-1'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0

NAME='STAR-Lite'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} \
--bs ${BS} --noise_type 'dist' --basemodel 'DeepLabV3'

for t in 1 2 3 4 5; do
  python compute_prototype.py -c configs/config_voc.json \
  -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
  --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step ${t} --bs ${BS} --noise_type 'dist' --basemodel 'DeepLabV3'
  python train_voc.py -c configs/config_voc.json \
  -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
  --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step ${t} --lr ${LR} --bs ${BS} --freeze_bn \
  --mem_size ${MEMORY_SIZE} --noise_type 'dist' --basemodel 'DeepLabV3'
done
