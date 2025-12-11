#!/bin/bash
#SBATCH --job-name="CPC_baseline"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --mem=501600
#SBATCH --gres=gpu:full:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

unset LD_LIBRARY_PATH  # load cuDNN intalled along with pytorch, rather than the cuDNN in Haicore system

source /hkfs/home/haicore/iwu/qa8171/env/HY3.11/bin/activate

Percent=1 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py
Percent=0.8 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py
Percent=0.6 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py
Percent=0.4 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py
Percent=0.2 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py
Percent=0.1 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py
Percent=0.05 python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune.py


# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_100per.py
# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_80per.py
# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_60per.py
# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_40per.py
# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_20per.py
# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_10per.py
# python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_fine_tune_5per.py