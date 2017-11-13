#!/usr/bin/env python
# -*- coding: utf-8 -*-

from opfsdr import opf
import json, re
import requests
import sys
testcases = {}
clist = []

# Retrieve list of MATPOWER test cases
response = requests.get('https://api.github.com/repos/MATPOWER/matpower/contents/data')
clist += json.loads(response.text)

# Retrieve list of pglib-opf test cases
response = requests.get('https://api.github.com/repos/power-grid-lib/pglib-opf/contents/')
clist += json.loads(response.text)
response = requests.get('https://api.github.com/repos/power-grid-lib/pglib-opf/contents/api')
clist += json.loads(response.text)
response = requests.get('https://api.github.com/repos/power-grid-lib/pglib-opf/contents/sad')
clist += json.loads(response.text)

# Build dictionary with test case URLs
for c in clist:
    if not c['name'].endswith('.m'): continue
    casename = c['name'].split('.')[0]
    testcases[casename] = c['download_url']
del clist

# Generate list of options
opts = {'truncate_gen_bounds': [50.0],
        'branch_rmin': [1e-5],
        'conversion': [True],
        'tfill':[16],
        'tsize':[0],
        'scale':[True,False]}

def expand_options(params):
    L = [[]]
    for k in params.keys():
        L = [li + [e] for li in L for e in params[k]]
    return [dict(zip(params.keys(),li)) for li in L]

# Build semidefinite relaxations and export in SeDuMi format (.mat)
for ckey,url in testcases.items():

    print("Exporting %s.." % (ckey))
    for options in expand_options(opts):
        try:
            prob = opf(url, **options)
        except TypeError:
            print("Failed to read %s." % ckey, file=sys.stderr)
            break
        except ValueError:
            print("ValueError: %s." % ckey, file=sys.stderr)
            break
        if options['scale']:
            scaled = '_scaled'
        else:
            scaled = ''
        fname = "%s_ts%i_tf%i%s.mat" % (ckey,options['tsize'],options['tfill'],scaled)
        prob.export(str(fname))
