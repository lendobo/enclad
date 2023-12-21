#!/bin/bash

#SBATCH --job-name=SYNTH_ARR
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err
#SBATCH -t 00:45:00 # Default time for the shortest tasks (100 and 200)
#SBATCH -N 5
#SBATCH --tasks-per-node 16 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com
#SBATCH --array=1-360%5 # Updated array range

module purge
module load 2022
module load R/4.2.1-foss-2022a
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate hnitr3

cd /home/mbarylli/thesis_code/Networks/

# Define arrays for p, n, and fp_fn values
p_values=(100 200 400) # Updated p_values
n_values=(50 100 250 500 1000 2000)
fp_fn_values=(0 0.25 0.5 0.75 1)

# Recalculate the total number of combinations
total_combinations=90 # 3 p_values * 6 n_values * 5 fp_fn_values

# Calculate the index for p, n, fp_fn, and seed based on SLURM_ARRAY_TASK_ID
seed_index=$(( (SLURM_ARRAY_TASK_ID - 1) / total_combinations ))
p_index=$(( ((SLURM_ARRAY_TASK_ID - 1) % total_combinations) / (6 * 5) ))
n_index=$(( (((SLURM_ARRAY_TASK_ID - 1) % total_combinations) % (6 * 5)) / 5 ))
fp_fn_index=$(( ((SLURM_ARRAY_TASK_ID - 1) % total_combinations) % 5 ))

# Get the values for p, n, fp_fn, and seed
P_VAL=${p_values[$p_index]}
N_VAL=${n_values[$n_index]}
FP_FN_VAL=${fp_fn_values[$fp_fn_index]}
SEED_VAL=$((seed_index + 1))

# Adjust the job time limit for p=400
if [ $P_VAL -eq 400 ]; then
    scontrol update JobId=$SLURM_JOB_ID TimeLimit=00:45:00
    scontrol update JobId=$SLURM_JOB_ID JobName="SYNTH_nu_pnf_p400"
fi

# Run the Python script with the mapped values of --p, --n, --fp_fn, and --seed
mpirun python piglasso.py --p $P_VAL --n $N_VAL --fp_fn $FP_FN_VAL --Q 2000 --b_perc 0.6 --llo 0.01 --lhi 0.5 --lamlen 100 --seed $SEED_VAL --dens 0.04

