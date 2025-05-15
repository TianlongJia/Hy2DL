#!/bin/bash
#SBATCH --job-name="lstm_camelsde"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_h100
#SBATCH --mem=760000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

source /pfs/data6/home/ka/ka_iwu/ka_qa8171/envs/HY3.9/bin/activate

python /pfs/data6/home/ka/ka_iwu/ka_qa8171/Project/Hy2DL/MFLSTM_hourly_de_TL_test.py