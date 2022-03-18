#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=tir-0-15,tir-1-28,tir-0-19,tir-1-7,tir-1-11,tir-1-23,tir-0-36,tir-0-32
#SBATCH --mem=50G
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=run_bird

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate venv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
python3 -m src.comp_datasets.bird_data