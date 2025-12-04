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

python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_100per_NoEmb.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_80per_NoEmb.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_60per_NoEmb.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_40per_NoEmb.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_20per_NoEmb.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_10per_NoEmb.py
python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/CPC_notebooks/CPC_baseline_5per_NoEmb.py