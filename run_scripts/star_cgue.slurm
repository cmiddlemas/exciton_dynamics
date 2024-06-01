#!/bin/bash
# https://unix.stackexchange.com/questions/285690/slurm-custom-standard-output-name
#SBATCH --job-name=star_cgue        # create a short name for your job
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=clairemiddlemas@princeton.edu

module purge
module load anaconda3/2024.2
conda activate qutip500

python3 -u star_cgue.py
