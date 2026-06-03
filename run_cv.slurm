#!/bin/bash
#SBATCH --job-name=nflow_cv
#SBATCH --account=project_2019415
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

module purge
module load tensorflow/2.14
# REMOVED: module load python-data/3.10 to prevent it from replacing TensorFlow

# It is fine to keep this pip command if the TensorFlow module is missing these!
#pip install --quiet --user scipy matplotlib pandas

# FIXED PATH: Points directly to where train_cv.py is located
cd /scratch/project_2019415

FILES=(
    "3D_data/observables_core_data1.dat"
    "3D_data/observables_core_data2.dat"
    "3D_data/observables_core_data3.dat"
    "3D_data/observables_cusp_data1.dat"
    "3D_data/observables_cusp_data2.dat"
    "3D_data/observables_cusp_data3.dat"
)

SELECTED_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo " SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo " Processing file  : ${SELECTED_FILE}"
echo " Node             : $(hostname)"
echo " Start time       : $(date)"
echo "============================================================"

export SLURM_FILE_IDX="${SLURM_ARRAY_TASK_ID}"

python train_cv.py

echo "============================================================"
echo " Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "============================================================"