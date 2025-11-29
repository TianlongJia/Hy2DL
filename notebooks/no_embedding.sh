#!/bin/bash
#SBATCH --job-name="no_embedding"
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --mem=501600
#SBATCH --gres=gpu:full:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tianlong.jia@kit.edu

unset LD_LIBRARY_PATH  # load cuDNN intalled along with pytorch, rather than the cuDNN in Haicore system

source /hkfs/home/haicore/iwu/qa8171/env/HY3.11/bin/activate


python /hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/notebooks/no_embedding.py