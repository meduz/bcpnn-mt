import os
import re

#to_match = '^LargeScaleModel_AIII_scaleLatency0\.15(.*)delayScale(\d+)$'
to_match = '^LargeScaleModel_AIII_scaleLatency0\.15(.*)tstim(\d+)$'
#to_match = '^LargeScaleModel_IIII_scaleLatency0\.15(.*)delayScale(\d+)$'
#to_match = '^LargeScaleModel_RRRR_(.*)'
#to_match = '^LargeScaleModel_AAAA_(.*)'
script_name = 'plot_prediction.py'


for thing in os.listdir('.'):
    m = re.search('%s' % to_match, thing)
    if m:
        cmd = 'python %s %s' % (script_name, thing)
        print cmd
        os.system(cmd)

