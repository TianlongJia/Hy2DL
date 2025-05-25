#!/bin/bash
#SBATCH --job-name="14_day_in_hourly"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --mem=501600
#SBATCH --gres=gpu:full:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

source /hkfs/home/haicore/iwu/qa8171/env/HY3.9/bin/activate

python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/14_day_2.py