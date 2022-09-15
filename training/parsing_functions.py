import os
import fileinput
import sys
import re
from datetime import datetime

def find_replace(file, find, replace):
    with fileinput.FileInput(file, inplace=True) as file:
        for line in file:
            print(line.replace(find, replace), end='')

def unix_command(*args):
    os.system(" ".join(list(args)))
