#! /usr/bin/env python

import os
import re
import sys
import subprocess
import time

def wait_for_processes(proc_array):
    for i in range(len(proc_array)):
        proc_array[i].wait()

if(len(sys.argv) == 1):
    alg = "GP"
else:
    alg = str(sys.argv[1])
    

if(alg == "GP" or alg == "SVM"):
    command = "$MRG_GP_ROOT/build/"+alg+"_active_learning"
else:
    command = "$MRG_GP_ROOT/build/cross_over"

config_file = "$MRG_GP_ROOT/config/SPINSPEC.conf"

n_outer = 10;
n_inner = 10;

for j in range(n_outer):
    proc_array = []
    for i in range(n_inner):
        dirname = "run" + "{0:04d}".format(i + 10*j)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            os.chdir(dirname)
            f = open('out.txt', 'w')
            p = subprocess.Popen([os.path.expandvars(command), \
                              os.path.expandvars(config_file)], stdout=f, stderr=f)
            proc_array.append(p)
            os.chdir('..')
    print "Running batch "+str(j)+" out of "+str(n_outer)+", with "+str(n_inner)+" processes"
    wait_for_processes(proc_array)
