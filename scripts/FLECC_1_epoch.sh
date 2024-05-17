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
echo "FLECC"
start_time=$(date +%s)
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py /data/ap421/EFE_Project/data/datasets/dataset_FLECC-2024-05-16.json 32 1 1e-4 /data/ap421/FLECC_fold_0_1.json /data/ap421/FLECC_fold_0_competitive 0
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
