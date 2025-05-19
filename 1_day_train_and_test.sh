#!/bin/bash
#SBATCH --job-name="train_and_test"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_h100
#SBATCH --mem=760000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

source /pfs/data6/home/ka/ka_iwu/ka_qa8171/envs/Hy2DL/bin/activate\

python /pfs/data6/home/ka/ka_iwu/ka_qa8171/Project/Hy2DL/1_day_train_and_test.py