#!/usr/bin/env python
# -*- coding: utf-8 -*-
from opfsdr import opf

case_url = 'https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case300.m'

# Construct and solve semidefinite relaxation
options = {'conversion': True, 'verbose':1}
prob = opf(case_url, **options)
sol = prob.solve(solver="mosek")
print("%80s" % (80*'-'))
print("Generation cost: %.2f USD/hour" % sol['cost'])
print("Minimum clique eigenvalue ratio: %.2e" % min(sol['eigratio']))
print("%80s" % (80*'-'))
