#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --output=op_file.txt
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
module load cuda/10.0
module add cuda/10.0


export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /scratch/
if [ ! -d aman.atman ]; then
    mkdir aman.atman
fi

cd aman.atman/
if [ ! -d tr ]; then
    mkdir spin
fi

cd spin/
# rm -r *
rsync -avz aman.atman@ada.iiit.ac.in:/home2/aman.atman/spin/ --exclude=logs ./

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate spin

python -m experiments.run_imputation --config imputation/spin_h.yaml --dataset-name la_point