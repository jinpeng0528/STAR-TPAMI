#!/bin/bash

GPU=0
BS=24
SAVEDIR='saved_city'

TASKSETTING='overlap'
TASKNAME='13-6'
INIT_LR=0.005
LR=0.0005
MEMORY_SIZE=0

NAME='STAR-Lite'
python train_city.py -c configs/config_city.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} \
--bs ${BS} --noise_type 'dist' --basemodel 'DeepLabV3'

python compute_prototype.py -c configs/config_city.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --bs ${BS} --noise_type 'dist' --basemodel 'DeepLabV3'
python train_city.py -c configs/config_city.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} --seed 826 \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE} \
--noise_type 'dist' --basemodel 'DeepLabV3'
