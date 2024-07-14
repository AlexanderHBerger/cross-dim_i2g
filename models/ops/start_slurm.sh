#!/bin/sh
#SBATCH --job-name=install
#SBATCH --output=/home/guests/alexander_berger/slurm_out/install.out
#SBATCH --error=/home/guests/alexander_berger/slurm_out/install.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --gres=gpu:1

# load python module
ml python/anaconda3

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this$
#conda activate relationformer
source /home/guests/alexander_berger/relationformer/env/bin/activate

cd /home/guests/alexander_berger/relationformer/models/ops
python setup.py install
