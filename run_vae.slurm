#!/bin/bash
#SBATCH --job-name=vae_exp
#SBATCH --chdir=/users/stud/knebelkr/masterarbeit
#SBATCH --output=logs/vae_%A_%a.out
#SBATCH --error=logs/vae_%A_%a.err
#SBATCH --array=0-23%4
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mail-user=knebelkr@hu-berlin.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# === Conda aktivieren ===
source /opt/conda/etc/profile.d/conda.sh
conda activate tf_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Projektpfade ===
BASE_DIR="/users/stud/knebelkr/masterarbeit"
PROJECT_DIR="${BASE_DIR}/VAE_1302"
LOG_DIR="${BASE_DIR}/logs"

# === Ordner anlegen (falls noch nicht vorhanden) ===
mkdir -p "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# Optional: falls du Repo-Inhalt in VAE_1302 erwartest, aber er fehlt, abbrechen mit Hinweis
if [ ! -f "${PROJECT_DIR}/src/main.py" ]; then
  echo "[ERROR] ${PROJECT_DIR}/src/main.py nicht gefunden."
  echo "Bitte stelle sicher, dass dein Repository nach ${PROJECT_DIR} kopiert/ausgecheckt wurde."
  exit 1
fi

# === Ins Projektverzeichnis wechseln ===
cd "${PROJECT_DIR}"

START="${SLURM_ARRAY_TASK_ID}"
END="${SLURM_ARRAY_TASK_ID}"

echo "--------------------------------------"
echo "Starte Experimente ${START}-${END}"
echo "Hostname: $(hostname)"
echo "SLURM Job-ID: ${SLURM_JOB_ID}, Array-Task: ${SLURM_ARRAY_TASK_ID}"
echo "Projekt: ${PROJECT_DIR}"
echo "--------------------------------------"

python -m src.main --range "${START}-${END}"

echo "--------------------------------------"
echo "Job ${SLURM_ARRAY_TASK_ID} abgeschlossen."
echo "--------------------------------------"
