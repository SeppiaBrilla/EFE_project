#!/bin/bash
# Job name:
#SBATCH --job-name=ap421TrainJob
#
# Partition:
#SBATCH --partition=gpu-serv-02-part
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=4
#
#SBATCH --mem=16GB
#
#SBATCH --gpus-per-task=1
#
#SBATCH --gpu-bind=single:1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s)
echo "fold 0"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_0_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_0_competitive --fold 0 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_0_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_0_competitive --fold 0 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_0_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_0_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_0_competitive --fold 0 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_0_competitive_final

echo "fold 1"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_1_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_1_competitive --fold 1 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_1_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_1_competitive --fold 1 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_1_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_l_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_1_competitive --fold 1 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_1_competitive_final

echo "fold 2"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_2_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_2_competitive --fold 2 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_2_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_2_competitive --fold 2 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_2_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_2_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_2_competitive --fold 2 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_2_competitive_final

echo "fold 3"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_3_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_3_competitive --fold 3 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_3_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_3_competitive --fold 3 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_3_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_3_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_3_competitive --fold 3 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_3_competitive_final

echo "fold 4"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_4_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_4_competitive --fold 4 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_4_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_4_competitive --fold 4 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_4_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_4_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_4_competitive --fold 4 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_4_competitive_final

echo "fold 5"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_5_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_5_competitive --fold 5 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_5_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_0_competitive --fold 5 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_5_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_5_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_5_competitive --fold 5 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_5_competitive_final

echo "fold 6"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_6_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_6_competitive --fold 6 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_6_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_6_competitive --fold 6 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_6_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_6_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_6_competitive --fold 6 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_6_competitive_final

echo "fold 7"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_7_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_7_competitive --fold 7 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_7_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_7_competitive --fold 7 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_7_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_7_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_7_competitive --fold 7 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_7_competitive_final

echo "fold 8"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_8_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_8_competitive --fold 8 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_8_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_8_competitive --fold 8 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_8_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_8_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_8_competitive --fold 8 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_8_competitive_final

echo "fold 9"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_9_mult_2.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_9_competitive --fold 9 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/AbnormalMagicHexagon/fold_9_mult_1_init.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_9_competitive --fold 9 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_9_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_AbnormalMagicHexagons-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/AbnormalMagicHexagon/fold_9_mult_1_fin.json --save /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_9_competitive --fold 9 --multiplier 1 --pre_trained /data/ap421/weights/AbnormalMagicHexagon/AbnormalMagicHexagons_fold_9_competitive_final
