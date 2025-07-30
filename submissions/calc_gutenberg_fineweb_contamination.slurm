#!/bin/bash
#SBATCH --account=root
#SBATCH --environment=ngc24-11
#SBATCH --job-name=gut_fineweb_conta
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/decont/gut_fineweb_conta_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/decont/gut_fineweb_conta_%j.err

# Load environment based on architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # ARM architecture
    source /users/xyixuan/miniconda3/etc/profile.d/conda.sh
    conda activate decont
else
    # x86 architecture
    source /users/xyixuan/miniconda3_x86/etc/profile.d/conda.sh
    conda activate decont
fi

# Set working directory
cd /iopsstor/scratch/cscs/xyixuan/PDM

echo "Starting Gutenberg to FineWeb contamination calculation"
echo "Time: $(date)"
echo "Node: $HOSTNAME"
echo "CPUs available: $(nproc)"

# No memory monitoring needed with optimized implementation

# Set paths
GUTENBERG_INDICES="/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/gutenberg"
FINEWEB_INDICES="/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/fineweb"
OUTPUT_FILE="/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/gutenberg_to_fineweb_contamination.json"

# Run the contamination calculation
echo "Starting contamination calculation..."
python -m src.decont.calc_gutenberg_to_fineweb_conta \
    --gutenberg-indices "$GUTENBERG_INDICES" \
    --fineweb-indices "$FINEWEB_INDICES" \
    --output "$OUTPUT_FILE" \
    --max-workers 64

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Contamination calculation completed successfully"
    echo "Results saved to: $OUTPUT_FILE"
    
    # Show summary of results
    echo -e "\n=== Results Summary ==="
    if [ -f "$OUTPUT_FILE" ]; then
        python -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
    for gut_name, results in data.items():
        if 'statistics' in results:
            stats = results['statistics']
            print(f'{gut_name}:')
            print(f'  Mean contamination: {stats[\"mean\"]*100:.2f}%')
            print(f'  Max contamination: {stats[\"max\"]*100:.2f}%')
            print(f'  Chunks checked: {results.get(\"chunks_processed\", \"N/A\")}')
        "
    fi
else
    echo "Contamination calculation failed with exit code $EXIT_CODE"
fi

echo -e "\nJob completed at: $(date)"