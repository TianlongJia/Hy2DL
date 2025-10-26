#!/bin/bash
#SBATCH --job-name="forecast_292_basins"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --mem=501600
#SBATCH --gres=gpu:full:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

source /hkfs/home/haicore/iwu/qa8171/env/HY3.11/bin/activate

python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/notebooks/LSTM_Forecast_DE.py

