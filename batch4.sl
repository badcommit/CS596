#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --output=log4.txt
#SBATCH -A anakano_429
#SBATCH --gres=gpu:v100:1
module purge
module spider python/3.9.2
eval "$(conda shell.bash hook)"
conda activate /project/anakano_429/zhangshe_test
mpiexec -n $SLURM_NTASKS python -u main.py