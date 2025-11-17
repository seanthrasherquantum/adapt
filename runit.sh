#!/bin/bash
#$ -N AVQ_main_run
#$ -cwd
#$ -j y
#$ -o results/logfiles/avq_main_$JOB_ID.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=10G
#$ -pe sharedmem 4

export OMP_NUM_THREADS=1
. /etc/profile.d/modules.sh
module load anaconda

# Explicitly initialize conda for non-interactive shell
source $(conda info --base)/etc/profile.d/conda.sh


# Activate your environment

# ...existing code...

# Activate your environment (use name or full path if present)
ENV_PATH=/exports/eddie/scratch/s2434746/conda_envs/adapt
if [ -d "$ENV_PATH" ]; then
  conda activate "$ENV_PATH"
else
  echo "Warning: conda env not found at $ENV_PATH — activating 'base' instead"
  conda activate base
fi

# Make sure the AVQ source directory is visible to Python (optional but safe)
export PYTHONPATH=/exports/eddie/scratch/s2434746/PhD/adapt:$PYTHONPATH

# Reinstall in editable mode inside the job (optional)
python -m pip install -e .
pip install pyscf
# ...existing code...

echo "=== Starting main.py at $(date) ==="


# Run the script with arguments
python /exports/eddie/scratch/s2434746/PhD/adapt/main.py $1 $2