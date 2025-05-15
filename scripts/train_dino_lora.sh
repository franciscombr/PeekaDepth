#!/bin/bash
#
#SBATCH --partition=debug_8gb   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=debug_8gb        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=ft_dino     # Job name
#SBATCH --output=./logs/slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=./logs/slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python3 src/train.py --config src/config/dinov2_conf_lora.yaml