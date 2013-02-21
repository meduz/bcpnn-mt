import os
import re

to_match = 'delayScale1'
script_name = 'plot_prediction.py'


for thing in os.listdir('.'):

    m = re.search('LargeScale(.*)%s$' % to_match, thing)
    if m:
        cmd = 'python %s %s' % (script_name, thing)
        print cmd
#        os.system(cmd)

