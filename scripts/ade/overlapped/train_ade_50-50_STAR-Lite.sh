#!/bin/bash

GPU=0,1
BS=12  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='50-50'
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0

NAME='STAR-Lite'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

for t in 1 2; do
  python compute_prototype.py -c configs/config_ade.json \
  -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
  --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step ${t} --bs ${BS} --noise_type 'dist' --basemodel 'DeepLabV3'
  python train_ade.py -c configs/config_ade.json \
  -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
  --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step ${t} --lr ${LR} --bs ${BS} --freeze_bn \
  --noise_type 'dist' --basemodel 'DeepLabV3'
done
