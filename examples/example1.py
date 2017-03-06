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

# Select test case with 300 busses
case = [case for case in testcases if case['busses'] == 300][0]

# Construct and solve semidefinite relaxation
options = {'branch_rmin': 1e-5, 'conversion': True, 'verbose':1}
prob = opf(case['url'], **options)
sol = prob.solve(solver="mosek")
print("%80s" % (80*'-'))
print("Generation cost: %.2f USD/hour" % sol['cost'])
print("Minimum clique eigenvalue ratio: %.2e" % min(sol['eigratio']))
print("%80s" % (80*'-'))
