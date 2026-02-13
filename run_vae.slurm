#!/bin/bash
#SBATCH --job-name=vae_exp
#SBATCH --chdir=/users/stud/knebelkr/masterarbeit/VAE_1302
#SBATCH --output=/users/stud/knebelkr/masterarbeit/VAE_1302/logs/vae_%A_%a.out
#SBATCH --error=/users/stud/knebelkr/masterarbeit/VAE_1302/logs/vae_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mail-user=knebelkr@hu-berlin.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Wichtig: Slurm will StdOut/StdErr öffnen, bevor dein Skript läuft.
# Daher muss der Ordner schon existieren. Das machen wir hier zusätzlich,
# aber du solltest ihn auch EINMAL manuell anlegen (siehe unten).
mkdir -p /users/stud/knebelkr/masterarbeit/VAE_1302/logs
mkdir -p /users/stud/knebelkr/masterarbeit/VAE_1302/runs

source /opt/conda/etc/profile.d/conda.sh
conda activate tf_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /users/stud/knebelkr/masterarbeit/VAE_1302

START="${SLURM_ARRAY_TASK_ID}"
END="${SLURM_ARRAY_TASK_ID}"

echo "--------------------------------------"
echo "Starte Experimente ${START}-${END}"
echo "Hostname: $(hostname)"
echo "SLURM Job-ID: ${SLURM_JOB_ID}, Array-Task: ${SLURM_ARRAY_TASK_ID}"
echo "PWD: $(pwd)"
echo "--------------------------------------"

python -m src.main --range "${START}-${END}"

echo "--------------------------------------"
echo "Job ${SLURM_ARRAY_TASK_ID} abgeschlossen."
echo "--------------------------------------"
