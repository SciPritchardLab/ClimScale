import sys
import wandb
from parsing_functions import *

sweep_id = sys.argv[1]
project_name = sys.argv[2]
runs_per_gpu = sys.argv[3]
partition = sys.argv[4]
clock_time = sys.argv[5]
# python train_models.py project_name runs_per_gpu partition clock_time

job_name = project_name + ": " + sweep_id

tuning_script = "tuning_script.py"
sbatch_script = "sbatch_script.sh"

unix_command("cp", "tuning_template.py", tuning_script)
unix_command("cp", "sbatch_template.sh", sbatch_script)
find_replace(tuning_script, "PROJECT_NAME_HERE", project_name)
find_replace(tuning_script, "RUNS_PER_GPU_HERE", runs_per_gpu)
find_replace(tuning_script, "SWEEP_ID_HERE", sweep_id)
find_replace(sbatch_script, "PARTITION_HERE", partition)
if partition == "GPU":
    find_replace(tuning_script, "NUM_GPUS_PER_NODE_HERE", "8")
    find_replace(sbatch_script, "NUM_GPUS_PER_NODE_HERE", "8")
    find_replace(sbatch_script, "NTASKS_HERE", "8")
elif partition == "GPU-shared":
    find_replace(tuning_script, "NUM_GPUS_PER_NODE_HERE", "2")
    find_replace(sbatch_script, "NUM_GPUS_PER_NODE_HERE", "2")
    find_replace(sbatch_script, "NTASKS_HERE", "2")
find_replace(sbatch_script, "JOB_NAME_HERE", job_name)
find_replace(sbatch_script, "CLOCK_TIME_HERE", clock_time)

# comment this out if testing
unix_command("sbatch", sbatch_script)

# comment these lines out if not testing
# unix_command("cat", sbatch_script)
# unix_command("rm", sbatch_script)
# unix_command("cat", "run-dynamic.shared.sh")
# unix_command("cat", tuning_script)
# unix_command("rm", tuning_script)
