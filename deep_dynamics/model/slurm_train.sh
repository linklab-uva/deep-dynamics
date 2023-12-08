#!/bin/bash

#SBATCH --job-name=samir
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
source /etc/profile.d/modules.sh

echo $SLURMD_NODENAME

module load gcc-11.2.0
module load cuda-toolkit-11.7.0

cd /u/jlc9wr/deep-dynamics/deep_dynamics/model/

python3 tune_hyperparameters.py "$@"
