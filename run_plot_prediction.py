import os
import re

script_name = 'plot_prediction.py'

#w_ee = 0.030
t_blank = 200
conn_code = 'AAAA'

#to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)delayScale20_tblank%d$' % (conn_code, t_blank)
to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)delayScale20_tblank%d$' % (conn_code, t_blank)

for thing in os.listdir('.'):
    m = re.search('%s' % to_match, thing)
    if m:
        cmd = 'python %s %s' % (script_name, thing)
        print cmd
        os.system(cmd)

