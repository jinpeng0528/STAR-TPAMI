#!/bin/bash

GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0

NAME='STAR-Lite'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python compute_prototype.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --bs ${BS} --noise_type 'dist' --basemodel 'DeepLabV3'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn \
--mem_size ${MEMORY_SIZE} --noise_type 'dist' --basemodel 'DeepLabV3'



