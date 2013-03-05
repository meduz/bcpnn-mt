import os
import re

conn_code = 'AIII'
to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)wee(.*)tblank(\d+)$' % (conn_code)
to_match = '^LargeScaleModel_(.*)$'
script_name = 'plot_prediction.py'


for thing in os.listdir('.'):
    m = re.search('%s' % to_match, thing)
    if m:
        cmd = 'python %s %s' % (script_name, thing)
        print cmd
        os.system(cmd)

