#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=00:01:00
#SBATCH --output=log.txt
#SBATCH -A anakano_429
#SBATCH --gres=gpu:v100:1
module purge
module spider python/3.9.2:q:q
eval "$(conda shell.bash hook)"
conda activate /project/anakano_429/zhangshe_test
mpiexec -n $SLURM_NTASKS python main.py