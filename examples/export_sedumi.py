#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, re
from urllib import urlopen
from opfsdr import opf

# Retrieve list of test cases from MATPOWER repository
response = urlopen('https://api.github.com/repos/MATPOWER/matpower/contents/data')
clist = json.loads(response.read())
testcases = []
for c in clist:
    casename = c['name'].split('.')[0]
    testcases.append({'name':casename,
                      'url': c['download_url'],
                      'busses': int(re.search('case[a-z\_]*([0-9]+)',casename).group(1))})

# Generate list of options    
opts = {'truncate_gen_bounds': [40.0],
        'branch_rmin': [1e-5],
        'conversion': [True],
        'tfill':[0,8,16,32],
        'tsize':[0,8,16,32],
        'scale':[True,False]}

def expand_options(params):
    L = [[]]
    for k in params.keys():        
        L = [li + [e] for li in L for e in params[k]]
    return [dict(zip(params.keys(),li)) for li in L]

# Build semidefinite relaxations and export in SeDuMi format (.mat)
for case in testcases:
    
    # Skip small test cases    
    # if case['busses'] <= 300:
    #    continue
        
    print("Exporting %s.." % (case['name'])) 
    for options in expand_options(opts):
        try:
            prob = opf(case['url'], **options)
        except TypeError:
            print("Failed to read %s." % case['name'])
            break
        if options['scale']: 
            scaled = '_scaled'
        else: 
            scaled = ''
        fname = "%s_ts%i_tf%i%s.mat" % (case['name'],options['tsize'],options['tfill'],scaled)
        prob.export(str(fname)) 

