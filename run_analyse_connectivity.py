import os
import re

#to_match = '^LargeScaleModel_AIII_scaleLatency0\.15(.*)delayScale(\d+)$'
to_match = '^LargeScaleModel_AIII_(.*)delayScale(\d+)$'
script_name = 'analyse_connectivity.py'

for thing in os.listdir('.'):

    m = re.search('%s' % to_match, thing)
    if m:
        cmd = 'python %s %s' % (script_name, thing)
        print cmd
        os.system(cmd)

