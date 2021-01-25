#!/bin/bash
#SBATCH -n 39
#SBATCH -N 1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --gres=gpu:4

export CUDA_VISIBLE_DEVICES=0,1,2,3

module load cuda/10.0
module load cudnn/7.6-cuda-10.0
module load python/3.8.3
source ~/venv/bin/activate
python add-digits.py > log
