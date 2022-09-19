import os
import fileinput
import sys
import re
import numpy as np

family = sys.argv[1]
start = int(sys.argv[2])
stop = int(sys.argv[3])

diff = stop-start
spacer = 10

remainder = diff%spacer
numsets = diff//spacer

slurmScript = "slurm_trim_template.sh"


def findreplace(file, find, replace):
    with fileinput.FileInput(file, inplace=True) as file:
        for line in file:
            print(line.replace(find, replace), end='')

def call_sbatch(first, last):
    first = str(first)
    last = str(last)
    slurmScriptNew = "sbatchmini_" + family + "_" + first + "_" + last
    os.system(" ".join(["cp", slurmScript, slurmScriptNew]))
    findreplace(slurmScriptNew, "VARIANT", family)
    findreplace(slurmScriptNew, "START", first)
    findreplace(slurmScriptNew, "STOP", last)
    os.system(" ".join(["sbatch", slurmScriptNew, family, str(first), str(last)]))


while diff > 0:
    mid = start + spacer - 1
    if diff < spacer:
        call_sbatch(start, start+remainder)
    else:
        call_sbatch(start, mid)
    start = start + spacer
    diff = diff - spacer


