#!/bin/bash
#SBATCH --job-name="lstm_camelsde"
#SBATCH --ntasks=1
#SBATCH --time=72
#SBATCH --partition=gpu_a100_il
#SBATCH --mem=510000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

source /pfs/data6/home/ka/ka_iwu/ka_qa8171/envs/HY3.9/bin/activate

python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/pred_1_14_day_BW.py