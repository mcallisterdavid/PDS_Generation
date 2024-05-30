import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
current_path = os.path.abspath(os.path.dirname(__file__))

# Initialize an empty list to store the commands
commands = []

# Loop through each line in the file
for shard_id in range(50):
    # Create the command with the extracted filename
    command = f'python3 baseline_run_coco.py --mode vsd --cfg_scale 7.5  --n_step 2500  --seed 0 --shard_id {shard_id}'
    # command = f'python3 baseline_run_coco.py --mode sds --cfg_scale 100  --n_step 2500 --lr 0.01 --seed 0 --shard_id {shard_id}'
    # command = f'python3 baseline_run_coco.py --mode nfsd --cfg_scale 7.5  --n_step 2500 --lr 0.01 --seed 0 --shard_id {shard_id} --init_image_method rand'
    # command = f'python3 baseline_run_coco.py --mode sds++ --cfg_scale 100  --n_step 200 --lr 0.1 --seed 0 --shard_id {shard_id}'
    # command = f'python3 ours_run_coco.py --config.shard_id {shard_id}'
    # command = f'python3 csd_run_coco.py --config.shard_id {shard_id}'
    # command = f'python3 baseline_run_coco.py --mode ddim --seed 0 --shard_id {shard_id}'
    
    JNAME = f"{shard_id}_ddim20_generation"
    SCRIPT = f"{current_path}/slurm/temp/run.{JNAME}.sh"
    SLURM = f"{current_path}/slurm/temp/run.{JNAME}.slrm"

    # Write to SCRIPT file
    with open(SCRIPT, 'w') as script_file:
        script_file.write("#!/bin/sh\n")
        script_file.write("{\n")
        script_file.write(
            command
        )
        script_file.write("\nkill -9 $$\n")
        script_file.write("} &\n")
        script_file.write("child_pid=$!\n")
        script_file.write('trap "echo \'TERM Signal received\';" TERM\n')
        script_file.write(
            'trap "echo \'Signal received\'; if [ \"$SLURM_PROCID\" -eq \"0\" ]; then sbatch '
            f'{SLURM}; fi; kill -9 $child_pid; " USR1\n'
        )
        script_file.write("while true; do sleep 1; done\n")

    # Write to SLURM file
    with open(SLURM, 'w') as slurm_file:
        slurm_file.write("#!/bin/sh\n")
        slurm_file.write(f"#SBATCH --job-name={JNAME}\n")
        slurm_file.write(
            f"#SBATCH --output=/fs/vulcan-projects/contrastive_learning_songweig/slurm/threestudio/coco/{JNAME}.out\n"
        )
        slurm_file.write(
            f"#SBATCH --error=/fs/vulcan-projects/contrastive_learning_songweig/slurm/threestudio/coco/{JNAME}.err\n"
        )
        slurm_file.write("#SBATCH --account=vulcan-jbhuang\n")
        slurm_file.write("#SBATCH --qos=vulcan-scavenger\n")
        slurm_file.write("#SBATCH --partition=vulcan-scavenger\n")
        # slurm_file.write("#SBATCH --account=scavenger\n")
        #slurm_file.write("#SBATCH --qos=scavenger\n")
        #slurm_file.write("#SBATCH --partition=scavenger\n")
        slurm_file.write("#SBATCH --gres=gpu:rtxa5000:1\n")
        slurm_file.write("#SBATCH --mem=32gb\n")
        slurm_file.write("#SBATCH --time=1-00:00:00\n")
        slurm_file.write("#SBATCH --nodes=1\n")
        slurm_file.write("#SBATCH --cpus-per-task=4\n")
        slurm_file.write("#SBATCH --ntasks-per-node=1\n")
        slurm_file.write(f"srun sh {SCRIPT}\n")

    # Submit the SLURM job
    os.system(f"sbatch {SLURM}")

# Print the generated commands
for cmd in commands:
    print(cmd)
