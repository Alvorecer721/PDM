#!/bin/bash
#SBATCH --account=root
#SBATCH --environment=data-pipeline
#SBATCH --job-name=gutenberg-index
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/gutenberg_index_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/gutenberg_index_%A.err

# Load environment based on architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    # Only activate conda on non-ARM architectures
    source /users/xyixuan/miniconda3/etc/profile.d/conda.sh
    conda activate decont
else
    echo "Running on ARM architecture ($ARCH), using SBATCH environment"
fi

# Set working directory
cd /iopsstor/scratch/cscs/xyixuan/PDM

echo "Starting Gutenberg index building"
echo "Time: $(date)"

# Build the index
python src/decont/build_gutenberg_index.py

echo -e "\nGutenberg index building completed at: $(date)"