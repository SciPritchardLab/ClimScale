import sys
from parsing_functions import *

project_name = sys.argv[1]
max_trials = sys.argv[2]
partition = sys.argv[3]
clock_time = sys.argv[4]
job_name = project_name + " " + str(max_trials)

tuning_script = "tuning_script.py"
sbatch_script = "sbatch_script.sh"

unix_command("cp", "tuning_template.py", tuning_script)
unix_command("cp", "sbatch_template.sh", sbatch_script)
find_replace(tuning_script, "PROJECT_NAME_HERE", project_name)
find_replace(tuning_script, "MAX_TRIALS_HERE", max_trials)
find_replace(sbatch_script, "PARTITION", partition)
if partition == "GPU":
    find_replace(tuning_script, "NUM_GPUS_PER_NODE_HERE", "8")
    find_replace(sbatch_script, "NTASKS_HERE", "9")
elif partition == "GPU-shared":
    find_replace(tuning_script, "NUM_GPUS_PER_NODE_HERE", "4")
    find_replace(sbatch_script, "NTASKS_HERE", "5")
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
