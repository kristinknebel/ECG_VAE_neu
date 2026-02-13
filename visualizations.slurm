#!/bin/bash
#SBATCH --job-name=vae_plots
#SBATCH --chdir=/users/stud/knebelkr/masterarbeit
#SBATCH --output=logs/vae_plots_%A.out
#SBATCH --error=logs/vae_plots_%A.err
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mail-user=knebelkr@hu-berlin.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# === Pfade ===
BASE_DIR="/users/stud/knebelkr/masterarbeit"
PROJECT_DIR="${BASE_DIR}/VAE_1302"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${LOG_DIR}"

# === Conda aktivieren ===
source /opt/conda/etc/profile.d/conda.sh
conda activate tf_env

# === Ins Projekt wechseln ===
cd "${PROJECT_DIR}"

# Wichtig: damit "python -m src.plot_from_run" src findet
export PYTHONPATH="${PROJECT_DIR}"

# Optional: Threads begrenzen (oft stabiler auf Cluster)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Parameter-Gitter ===
TSNE_POINTS=(3000 5000 8000)
TSNE_PERP=(20 30 40)
UMAP_NEIGH=(5 15 30)
UMAP_DIST=(0.1 0.5)

# === Welche Runs plotten? ===
# Trage hier deine VAE-Run-Ordner ein (relative Pfade ab PROJECT_DIR)
RUN_DIRS=(
  "runs/20260213_122952_973_p3608263__L16_B1_0_LR1e-04_A32"
  "runs/20260213_122955_525_p3220967__L16_B1_0_LR5e-04_A32"
  "runs/20260213_122955_553_p3117114__L32_B1_0_LR1e-04_A32"
  "runs/20260213_122955_584_p3067316__L32_B1_0_LR5e-04_A32"
)

for rd in "${RUN_DIRS[@]}"; do
  [ -d "$rd" ] || { echo "[WARN] missing: $rd"; continue; }

  for pts in "${TSNE_POINTS[@]}"; do
    for perp in "${TSNE_PERP[@]}"; do
      for nn in "${UMAP_NEIGH[@]}"; do
        for md in "${UMAP_DIST[@]}"; do

          OUT_DIR="$rd/plots_grid/tsne_pts${pts}_perp${perp}__umap_nn${nn}_md${md}"
          mkdir -p "$OUT_DIR"

          echo "=== $rd | pts=$pts perp=$perp nn=$nn md=$md ==="

          python -m src.plot_from_run \
            --run_dir "$rd" \
            --out_dir "$OUT_DIR" \
            --do_tsne --do_umap --dims both \
            --tsne_points "$pts" \
            --tsne_perplexity "$perp" \
            --umap_neighbors "$nn" \
            --umap_min_dist "$md" \
            --only_single \
            --per_class 200

        done
      done
    done
  done
done

echo "[OK] Plot-Job abgeschlossen."
