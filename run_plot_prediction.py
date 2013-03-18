import os
import re
import json
import sys

script_name = 'plot_prediction.py'


def run_by_match(to_match):

    for thing in os.listdir('.'):
        m = re.search('%s' % to_match, thing)
        if m:
            cmd = 'python %s %s' % (script_name, thing)
            print cmd
            os.system(cmd)

def get_filenames(fn_with_missing_data):
    f = file(fn_with_missing_data, 'r')
    list_of_dirs = json.load(f)
    print '\nlist_of_dirs', list_of_dirs
    for dir_name in list_of_dirs:
        cmd = 'python %s %s' % (script_name, dir_name)
        os.system(cmd)


if len(sys.argv) > 1:
    fn_in = sys.argv[1]
    # 'missing_data_dirs.json'
    get_filenames(fn_in)




conn_code = 'AIII'
to_match = '^LargeScaleModel_(.*)'
#to_match = '^LargeScaleModel_AIII_bx1.00e-01_bv1.00e-01_wsigmax2.50e-01_wsigmav2.50e-01_wee(.*)'
run_by_match(to_match)
