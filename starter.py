#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just for starting a pythonprogramm with iteration over all files in a folder.
Tobias Machnitzki
"""

import glob
import os
import sys

working_dir = "/data/mpi/mpiaes/obs/ACPC/allsky/m1705"
program = "ASCA_nc.py"

exit_now = False

for element in sorted(os.listdir(working_dir)):
    if exit_now == True:
        sys.exit()

    date = element
    print(date)
    os.system("python " + program + " -t " + date)
