import sys
from parsing_functions import *

project_name = sys.argv[1]
max_trials = sys.argv[2]

tuning_script = "tuning_template_test.py"
sbatch_script = "sbatch_template_test.sh"

unix_command("cp", "tuning_template.py", tuning_script)
unix_command("cp", "sbatch_template.sh", sbatch_script)
find_replace(tuning_script, "PROJECT_NAME_HERE", project_name)
find_replace(tuning_script, "MAX_TRIALS_HERE", max_trials)

# comment these lines out if not testing
unix_command("cat", tuning_script)
unix_command("rm", tuning_script)
unix_command("cat", sbatch_script)
unix_command("rm", sbatch_script)
