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

python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_1_day.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_2_day.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_4_day.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_7_day.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_14_day.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_28_day.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/Exp6_50_day.py
