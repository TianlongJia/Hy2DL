#!/bin/bash
#SBATCH --job-name="1_day_in_hourly"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --mem=501600
#SBATCH --gres=gpu:full:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

source /hkfs/home/haicore/iwu/qa8171/env/HY3.9/bin/activate

SEED=110 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py
SEED=111 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py
SEED=222 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py
SEED=333 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py 
SEED=444 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py 
# SEED=555 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py 
# SEED=666 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py 
# SEED=777 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py 
# SEED=888 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py 
# SEED=999 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/120_days.py
