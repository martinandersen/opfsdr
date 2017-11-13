#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semidefinite Relaxation of Optimal Power Flow Problems

Copyright 2017: Martin S. Andersen (martin.skovgaard.andersen@gmail.com)
License: GPL-3
"""

from cvxopt import sparse, spmatrix, spdiag, matrix, max, mul, div, exp, sqrt, solvers, lapack, blas, msk
from itertools import chain, compress
from math import pi, atan2, cos, sin, tan
import pickle, re
import numpy as np
import sys
import requests

try:
    import chompack
    __chompack__ = True
except ImportError:
    __chompack__ = False
try:
    from scipy.io import savemat
    from scipy.sparse import csr_matrix
    __scipy__ = True
except ImportError:
    __scipy__ = False


def _load_case(mfile, verbose = 0):
   """
   Imports data from Matpower case file (MATLAB m-file).
   """
   # Read m-file and strip MATLAB comments
   if mfile.startswith('http'):
      if verbose: print("Downloading case file: %s." % (mfile))
      response = requests.get(mfile)
      lines = response.text.split('\n')
   else:
      if verbose: print("Reading case file: %s." % (mfile))
      with open(mfile,"r") as f:
         lines = f.readlines()

   for k in range(len(lines)): lines[k] = lines[k].split('%')[0]
   case_as_str = "\n".join(lines)

   def str_to_array(s):
      return np.array([[float(v) for v in r.strip().split()] for r in s.strip(';\n\t ').split(';')])

   try:
      baseMVA = re.search("mpc.baseMVA = (\d+)", case_as_str).group(1)
      version = re.search("mpc.version = '(\d+)'", case_as_str).group(1)
      bus_str = re.search("mpc.bus = \[([-\s0-9e.;]+)\]", case_as_str).group(1)
      gen_str = re.search("mpc.gen = \[([-\s0-9e.;]+)\]", case_as_str).group(1)
      branch_str = re.search("mpc.branch = \[([-\s0-9e.;]+)\]", case_as_str).group(1)
      gencost_str = re.search("mpc.gencost = \[([-\s0-9e.;]+)\]", case_as_str).group(1)
   except:
      raise TypeError("Failed to parse case file.")
   else:
      if re.search('mpc.branch\(',case_as_str) or re.search('mpc.bus\(',case_as_str) or re.search('mpc.gen\(',case_as_str):
          raise TypeError("Case file not supported.")
      return {'baseMVA':float(baseMVA),
               'version':version,
               'bus':str_to_array(bus_str),
               'gen':str_to_array(gen_str),
               'gencost':str_to_array(gencost_str),
               'branch':str_to_array(branch_str)}

def _conelp_to_real(P, inplace = True, **kwargs):
    """
    Convert complex-valued cone LP to real-valued cone LP.
    """
    c,G,h,dims = P.problem_data
    offset_s = dims['l'] + sum(dims['q'])
    Glist, hlist = [G[:offset_s,:].real()],[h[:offset_s].real()]
    Gs = G[offset_s:,:]
    hs = sparse(h[offset_s:])
    ns = dims['s']

    if max(ns) <= 500:
        ri = []
        for k,si in enumerate(dims['s']):
            for j in range(si):
                for i in range(si):
                    ri.append((k,i,j))
        def blk_entry(idx):
            return ri[idx]
    else:
        def blk_entry(idx):
            blk = 0
            while idx >= ns[blk]**2:
                idx -= ns[blk]**2
                blk += 1
            return blk, idx % ns[blk], idx // ns[blk]

    offsets = [0]
    for ni in ns: offsets.append(offsets[-1] + (2*ni)**2)

    I,J,V = [],[],[]
    GI,GJ,GV = Gs.I,Gs.J,Gs.V
    for k in range(len(GI)):
        #blk,i,j = ri[GI[k]]
        blk,i,j = blk_entry(GI[k])
        ni = 2*ns[blk]
        if i == j:
            I.append([offsets[blk]+ni*j+i,offsets[blk]+ni*(ns[blk]+j)+ns[blk]+i])
            J.append(2*[GJ[k]])
            V.append(2*[0.5*GV[k].real])
        else:
            I.append([offsets[blk]+ni*j+i,offsets[blk]+ni*j+ns[blk]+i,\
                      offsets[blk]+ni*(ns[blk]+j)+i,offsets[blk]+ni*(ns[blk]+j)+ns[blk]+i])
            J.append(4*[GJ[k]])
            V.append([0.5*GV[k].real, 0.5*GV[k].imag, -0.5*GV[k].imag, 0.5*GV[k].real])

    Gr = spmatrix([v for v in chain(*V)],[i for i in chain(*I)],[j for j in chain(*J)],(offsets[-1],Gs.size[1]))

    I,J,V = [],[],[]
    hV, hI = hs.V,hs.I
    for k in range(len(hV)):
        #blk,i,j = ri[hI[k]]
        blk,i,j = blk_entry(GI[k])
        ni = 2*ns[blk]
        if i == j:
            I.append([offsets[blk]+ni*j+i,offsets[blk]+ni*(ns[blk]+j)+ns[blk]+i])
            J.append(2*[0])
            V.append(2*[0.5*hV[k].real])
        else:
            I.append([offsets[blk]+ni*j+i,offsets[blk]+ni*j+ns[blk]+i,\
                      offsets[blk]+ni*(ns[blk]+j)+i,offsets[blk]+ni*(ns[blk]+j)+ns[blk]+i])
            J.append(4*[0])
            V.append([0.5*hV[k].real, 0.5*hV[k].imag, -0.5*hV[k].imag, 0.5*hV[k].real])

    hr = spmatrix([v for v in chain(*V)],[i for i in chain(*I)],[j for j in chain(*J)],(offsets[-1],1))

    Glist += [Gr]
    hlist += [hr]
    G = sparse(Glist)
    h = sparse(hlist)
    dims = {'l':dims['l'],'q':dims['q'],'s':[2*si for si in dims['s']]}

    if inplace:
        P.problem_data = (c,G,h,dims)
        P.as_real = True
        P.Nx = P.Nx*2
    else:
        return c,G,h,dims

def _conelp_convert(P, **kwargs):
    """
    Chordal conversion.
    """
    tsize = kwargs.get('tsize',0)
    tfill = kwargs.get('tfill',0)
    inplace = kwargs.get('inplace',True)
    coupling = kwargs.get('coupling','full')

    prob = P.problem_data
    mf = chompack.merge_size_fill(tsize,tfill)
    cprob, b2s, symbs = chompack.convert_conelp(*prob,\
                                         coupling = coupling,\
                                         merge_function = mf)
    if inplace:
        P.problem_data = cprob
        P.blocks_to_sparse = b2s
    return cprob

def _conelp_scale(P, inplace = True, **kwargs):
    """
    Scale cone LP
    """
    inplace = kwargs.get('inplace', True)
    c,G,h,dims = P.problem_data
    cp = G.CCS[0]
    V = abs(G.V)
    u = matrix([max(V[cp[i]:cp[i+1]]) for i in range(len(cp)-1)])
    u = max(u,abs(c))
    G = G*spmatrix(div(1.0,u),range(len(u)),range(len(u)))
    c = div(c,u)
    nrm2h = blas.nrm2(h)
    if inplace: P.cost_scale *= max(1.0,nrm2h)
    if abs(nrm2h) > 1.0:  h /= nrm2h
    if inplace: P.problem_data = (c,G,h,dims)
    return c,G,h,dims

class opf(object):
    """
    Optimal power flow problem
    """
    def __init__(self, casefile, **kwargs):
        """
        Optional keyword arguments:

        branch_rmin         (default: -inf  )
        shunt_gmin          (default: -inf  )
        gen_elim            (default: 0.0   )
        truncate_gen_bounds (default: None  )
        line_constraints    (default: True  )
        scale               (default: False )
        """
        self.to_real = kwargs.get('to_real', True)
        self.conversion = kwargs.get('conversion', False)
        self.scale = kwargs.get('scale', False)
        self.shunt_gmin = kwargs.get('shunt_gmin', -float('inf'))
        self.branch_rmin = kwargs.get('branch_rmin', -float('inf'))
        self.gen_elim = kwargs.get('gen_elim', 0.0)
        self.truncate_gen_bounds = kwargs.get('truncate_gen_bounds',None)
        self.line_constraints = kwargs.get('line_constraints', True)
        self.pad_constraints = kwargs.get('pad_constraints', True)
        self.__verbose = kwargs.get('verbose',0)
        self.eigtol = kwargs.get('eigtol',1e5)

        ### load data
        data = _load_case(casefile, verbose = kwargs.get('verbose',0))
        assert data['version'] == '2'

        if kwargs.get('verbose',0): print("Extracting data from case file.")

        ### add data to object
        self.baseMVA = data['baseMVA']
        self.cost_scale = self.baseMVA
        self.nbus = data['bus'].shape[0]

        # branches in service
        active_branches = [i for i in range(data['branch'].shape[0]) if data['branch'][i,10] > 0]
        self.nbranch = len(active_branches)

        # generators in service
        active_generators = [i for i in range(data['gen'].shape[0]) if data['gen'][i,7] > 0]
        self.ngen = len(active_generators)

        # bus data
        self.busses = []
        for k in range(self.nbus):
            bus = {'id': int(data['bus'][k,0]),
                   'type': int(data['bus'][k,1]),
                   'Pd': data['bus'][k,2] / self.baseMVA,
                   'Qd': data['bus'][k,3] / self.baseMVA,
                   'Gs': data['bus'][k,4],
                   'Bs': data['bus'][k,5],
                   'area': data['bus'][k,6],
                   'Vm': data['bus'][k,7],
                   'Va': data['bus'][k,8],
                   'baseKV': data['bus'][k,9],
                   'maxVm': data['bus'][k,11],
                   'minVm': data['bus'][k,12]}
            if bus['Gs'] < self.shunt_gmin: bus['Gs'] = self.shunt_gmin
            self.busses.append(bus)

        self.bus_id_to_index = {}
        for k, bus in enumerate(self.busses):
            self.bus_id_to_index[bus['id']] = k

        # generator data
        self.generators = []
        ii_p = 0
        ii_q = 0
        for k in active_generators:
            gen = {'bus_id': int(data['gen'][k,0]),
                   'Pg': data['gen'][k,1] / self.baseMVA,
                   'Pmax': data['gen'][k,8] / self.baseMVA,
                   'Pmin': data['gen'][k,9] / self.baseMVA,
                   'Qg': data['gen'][k,2] / self.baseMVA,
                   'Qmax': data['gen'][k,3] / self.baseMVA,
                   'Qmin': data['gen'][k,4] / self.baseMVA,
                   'Vg': data['gen'][k,5],
                   'mBase': data['gen'][k,6]
                   }
            if gen['Pmax'] > gen['Pmin'] + self.gen_elim:
                gen['pslack'] = ii_p
                ii_p += 1
            else:
                # eliminate slack variable and set Pmin and Pmax to their average
                gen['pslack'] = None
                gen['Pmin'] = (gen['Pmax'] + gen['Pmin'])/2.0
                gen['Pmax'] = gen['Pmin']

            if gen['Qmax'] > gen['Qmin'] + self.gen_elim:
                gen['qslack'] = ii_q
                ii_q += 1
            else:
                # eliminate slack variable and set Qmin and Qmax to their average
                gen['qslack'] = None
                gen['Qmin'] = (gen['Qmax'] + gen['Qmin'])/2.0
                gen['Qmax'] = gen['Qmin']

            if self.truncate_gen_bounds:
                if gen['Pmin'] < -self.truncate_gen_bounds or gen['Pmax'] > self.truncate_gen_bounds:
                    if kwargs.get('verbose',0):
                       print("Warning: generator at bus %i with large active bound(s); decreasing bound(s)"%(gen['bus_id']))
                    gen['Pmin'] = max(-self.truncate_gen_bounds, gen['Pmin'])
                    gen['Pmax'] = min( self.truncate_gen_bounds, gen['Pmax'])

                if gen['Qmin'] < -self.truncate_gen_bounds or gen['Qmax'] > self.truncate_gen_bounds:
                    if kwargs.get('verbose',0):
                       print("Warning: generator at bus %i with large reactive bound(s); decreasing bound(s)"%(gen['bus_id']))
                    gen['Qmin'] = max(-self.truncate_gen_bounds, gen['Qmin'])
                    gen['Qmax'] = min( self.truncate_gen_bounds, gen['Qmax'])

            self.generators.append(gen)

        self.bus_id_to_genlist = {}
        for k, gen in enumerate(self.generators):
            if gen['bus_id'] in self.bus_id_to_genlist:
                self.bus_id_to_genlist[gen['bus_id']].append(k)
            else:
                self.bus_id_to_genlist[gen['bus_id']] = [k]

        # branch data
        self.branches = []
        for k in active_branches:
            branch = {'from': int(data['branch'][k,0]),
                      'to': int(data['branch'][k,1]),
                      'r': data['branch'][k,2],
                      'x': data['branch'][k,3],
                      'b': data['branch'][k,4],
                      'rateA': data['branch'][k,5],
                      'rateB': data['branch'][k,6],
                      'rateC': data['branch'][k,7],
                      'ratio': data['branch'][k,8],
                      'angle': data['branch'][k,9],
                      'angle_min': data['branch'][k,11],
                      'angle_max': data['branch'][k,12]
                      }
            if branch['r'] < self.branch_rmin:
                if kwargs.get('verbose',0): print("Warning: branch (%i:%i->%i) with small resistance; enforcing min. resistance"%(k,branch['from'],branch['to']))
                branch['r'] = self.branch_rmin
            if not self.line_constraints:
               branch['rateA'] = 0.0
            if not self.pad_constraints:
               branch['angle_min'] = -360.0
               branch['angle_max'] = 360.0
            elif (branch['angle_min'] > -360.0 and branch['angle_min'] <= -180.0) or (branch['angle_max'] < 360.0 and branch['angle_max'] >= 180.0):
               if kwargs.get('verbose',0): print("Warning: branch (%i:%i->%i) with unsupported phase angle diff. constraint; dropping constraint"%(k,branch['from'],branch['to']))
               branch['angle_min'] = -360.0
               branch['angle_max'] = 360.0

            self.branches.append(branch)


        # gen cost
        for i, k in enumerate(active_generators):
            gencost = {'model': int(data['gencost'][k,0]),
                       'startup': data['gencost'][k,1],
                       'shutdown': data['gencost'][k,2],
                       'ncoef': int(data['gencost'][k,3]),
                       'coef': data['gencost'][k,4:].T
                       }
            self.generators[i]['Pcost'] = gencost

        if data['gencost'].shape[0] == 2*data['gen'].shape[0]:
            offset = data['gen'].shape[0]
            for i, k in enumerate(active_generators):
                gencost = {'model': int(data['gencost'][offset + k,0]),
                           'startup': data['gencost'][offset + k,1],
                           'shutdown': data['gencost'][offset + k,2],
                           'ncoef': int(data['gencost'][offset + k,3]),
                           'coef': data['gencost'][offset + k,4:].T
                           }
                self.generators[i]['Qcost'] = gencost

        ### Compute bus admittance matrix and connection matrices
        j = complex(0.0,1.0)
        r = matrix([branch['r'] for branch in self.branches])
        b = matrix([branch['b'] for branch in self.branches])
        x = matrix([branch['x'] for branch in self.branches])
        Gs = matrix([bus['Gs'] for bus in self.busses])
        Bs = matrix([bus['Bs'] for bus in self.busses])

        tap = matrix([1.0 if branch['ratio'] == 0.0 else branch['ratio'] for branch in self.branches])
        angle = matrix([branch['angle'] for branch in self.branches])
        tap = mul(tap, exp(j*pi/180.*angle))
        self.tap = tap

        Ys = div(1.0, r + j*x)
        Ytt = Ys + j*b/2.0
        Yff = div(Ytt, mul(tap, tap.H.T))
        Yft = -div(Ys, tap.H.T)
        Ytf = -div(Ys, tap)
        Ysh = (Gs+j*Bs)/self.baseMVA

        self.Ybr = []
        for k in range(self.nbranch): self.Ybr.append([[Yff[k],Yft[k]],[Ytf[k],Ytt[k]]])

        f = matrix([self.bus_id_to_index[branch['from']] for branch in self.branches])
        t = matrix([self.bus_id_to_index[branch['to']] for branch in self.branches])

        Cf = spmatrix(1.0, range(self.nbranch), f, (self.nbranch,self.nbus))
        Ct = spmatrix(1.0, range(self.nbranch), t, (self.nbranch,self.nbus))
        Yf = spmatrix(matrix([Yff,Yft]), 2*list(range(self.nbranch)), matrix([f,t]), (self.nbranch,self.nbus))
        Yt = spmatrix(matrix([Ytf,Ytt]), 2*list(range(self.nbranch)), matrix([f,t]), (self.nbranch,self.nbus))
        Ybus = Cf.T*Yf + Ct.T*Yt + spmatrix(Ysh, range(self.nbus), range(self.nbus), (self.nbus,self.nbus))

        self.Cf = Cf
        self.Ct = Ct
        self.Yf = Yf
        self.Yt = Yt
        self.Ybus = Ybus

        if kwargs.get('verbose',0): print("Building cone LP.")
        self._build_conelp()
        if self.conversion:
           if kwargs.get('verbose',0):
               if kwargs.get('coupling','full') == 'full':
                  print("Applying chordal conversion to cone LP.")
               else:
                  print("Applying partial chordal conversion to cone LP.")
           _conelp_convert(self, **kwargs)
        if self.scale:
           if kwargs.get('verbose',0): print("Scaling cone LP.")
           _conelp_scale(self, **kwargs)
        if self.to_real:
           if kwargs.get('verbose',0): print("Converting to real-valued cone LP.")
           _conelp_to_real(self, **kwargs)

        return

    def pq_busses(self):
        return [bus for bus in self.busses if bus['type'] == 1]

    def pv_busses(self):
        return [bus for bus in self.busses if bus['type'] == 2]

    def ref_busses(self):
        return [bus for bus in self.busses if bus['type'] == 3]

    def iso_busses(self):
        return [bus for bus in self.busses if bus['type'] == 4]

    def busses_with_gen(self):
        return [bus for bus in self.busses if bus['id'] in self.bus_id_to_genlist]

    def busses_without_gen(self):
        return [bus for bus in self.busses if not bus['id'] in self.bus_id_to_genlist]

    def symmetric_branches(self):
        return [branch for branch in self.branches if branch['ratio'] == 0.0]

    def asymmetric_branches(self):
        return [branch for branch in self.branches if branch['ratio'] > 0.0]

    def branches_with_flow_constraints(self):
        return [(k,branch) for k,branch in enumerate(self.branches) if branch['rateA'] < 9900.0 and branch['rateA'] > 0]

    def branches_with_pad_constraints(self):
        return [(k,branch) for k,branch in enumerate(self.branches) if branch['angle_min'] > -360.0 and branch['angle_max'] < 360.0]

    def generators_with_fixed_cost(self):
      return [gen for gen in self.generators if gen['pslack'] is None]

    def generators_with_pwl_cost(self):
        return [gen for gen in self.generators if gen['Pcost']['model'] == 1]

    def generators_with_poly_cost(self):
        return [gen for gen in self.generators if gen['Pcost']['model'] == 2]

    def generators_with_var_real_power_and_linear_cost(self):
        return [gen for gen in self.generators if gen['Pcost']['model'] == 2 and gen['Pcost']['ncoef'] == 3 and gen['Pcost']['coef'][0] == 0.0 and gen['pslack'] is not None] + \
            [gen for gen in self.generators if gen['Pcost']['model'] == 2 and gen['Pcost']['ncoef'] == 2 and gen['pslack'] is not None]

    def generators_with_var_real_power_and_quadratic_cost(self):
        return [gen for gen in self.generators if gen['Pcost']['model'] == 2 and gen['Pcost']['ncoef'] == 3 and gen['Pcost']['coef'][0] > 0.0 and gen['pslack'] is not None]

    def generators_with_var_real_power(self):
        return [gen for gen in self.generators if gen['pslack'] is not None]

    def generators_with_var_reactive_power(self):
        return [gen for gen in self.generators if gen['qslack'] is not None]

    def generators_with_reactive_cost(self):
        return [gen for gen in self.generators if 'Qcost' in gen]

    def __str__(self):
        Nbranch_constr = len(self.branches_with_flow_constraints())
        Npad_constr = len(self.branches_with_pad_constraints())
        Ngen_lin = len(self.generators_with_var_real_power_and_linear_cost())
        Ngen_quad = len(self.generators_with_var_real_power_and_quadratic_cost())
        Ngen_fixed = len(self.generators_with_fixed_cost())
        Ntran = len(self.asymmetric_branches())
        return """Optimal power flow problem
* busses             : %6i
* generators         : %6i
   -  const. gen.    : %6i
   -  lin. cost      : %6i
   -  quad. cost     : %6i
* branches           : %6i
   -  flow constr.   : %6i
   -  phase constr.  : %6i
   -  transformer    : %6i"""\
    % (self.nbus,self.ngen,Ngen_fixed,Ngen_lin,Ngen_quad,self.nbranch,Nbranch_constr,Npad_constr,Ntran)

    def _build_conelp(self):
        Nx = self.nbus
        self.Nx = Nx
        nflow_constr = len(self.branches_with_flow_constraints())
        npad_constr = len(self.branches_with_pad_constraints())
        ngen_var_p = len(self.generators_with_var_real_power())
        ngen_qcost = len(self.generators_with_var_real_power_and_quadratic_cost())
        ngen_lcost = len(self.generators_with_var_real_power_and_linear_cost())
        self._ngen_var_p = ngen_var_p
        ngen_var_q = len(self.generators_with_var_reactive_power())
        self._ngen_var_q = ngen_var_q

        dims = {}
        dims['l'] = 2*self.nbus + 2*ngen_var_p + 2*ngen_var_q + ngen_qcost + 2*npad_constr
        dims['q'] = 2*nflow_constr*[3]
        dims['q'] += ngen_qcost*[3]
        dims['s'] = [Nx]

        offset = {}
        offset['t'] = 0                             # t   = aux. vars for epigraph formulation of quad. gen. power cost
        offset['wpl'] = offset['t'] + ngen_qcost    # wpl = slack lower bnd: Pmin[i] + wpl[i] = Pg[i]
        offset['wpu'] = offset['wpl'] + ngen_var_p  # wpu = slack upper bnd: Pg[i] + wpu[i] = Pmax[i]
        offset['wql'] = offset['wpu'] + ngen_var_p  # wql = slack lower bnd: Qmin[i] + wql[i] = Qg[i]
        offset['wqu'] = offset['wql'] + ngen_var_q  # wqu = slack upper bnd: Qg[i] + wqu[i] = Qmax[i]
        offset['ul'] = offset['wqu'] + ngen_var_q   # ul  = slack lower bnd: Vmin[i]**2 + ul[i] = abs(V[i])
        offset['uu'] = offset['ul'] + self.nbus     # uu  = slack upper bnd: abs(V[i]) + uu[i] = Vmax[i]**2
        offset['lpad'] = offset['uu'] + self.nbus
        offset['upad'] = offset['lpad'] + npad_constr
        offset['z'] = offset['upad'] + npad_constr  # z   = line flow const: z[k*3:(k+1)*3] in SOC of dim 3
        offset['w'] =  offset['z'] + sum(dims['q'])-3*ngen_qcost  # w  = epigraph of quad. gen cost: w[k*3:(k+1)*3] in SOC of dim 3
        offset['X'] = offset['w'] + 3*ngen_qcost                  # X  = SDR of X = V*V.H
        N = offset['X'] + Nx**2
        self.offset = offset

        dual_offset = [0]
        # power balance
        dual_offset.append(dual_offset[-1] + 2*self.ngen)
        # gen. limits (real)
        dual_offset.append(dual_offset[-1] + len(self.generators_with_var_real_power()))
        # gen. limits (reactive)
        dual_offset.append(dual_offset[-1] + len(self.generators_with_var_reactive_power()))
        # voltage constraints
        dual_offset.append(dual_offset[-1] + 2*self.nbus)
        # line constraints
        dual_offset.append(dual_offset[-1] + 6*len(self.branches_with_flow_constraints()))
        # pad pad_constraints
        dual_offset.append(dual_offset[-1] + 2*npad_constr)
        # quad. cost
        dual_offset.append(dual_offset[-1] + 3*len(self.generators_with_var_real_power_and_quadratic_cost()))

        self.dual_offset = matrix(dual_offset)

        ##
        ## Build h and initialize c
        ##
        h1 = spmatrix(1.0,range(ngen_qcost),ngen_qcost*[0],(N,1))  # quadratic cost
        I,V = [],[]
        for k,gen in enumerate(self. generators_with_var_real_power()):
            beta = gen['Pcost']['coef'][-2]
            if gen['Pcost']['ncoef'] > 2: beta += 2.0*self.baseMVA*gen['Pcost']['coef'][-3]*gen['Pmin']
            I.append(offset['wpl'] + gen['pslack'])
            V.append(beta)

        h2 = spmatrix(V,I,len(I)*[0],(N,1))
        h = h1 + h2
        c = []

        ##
        ## Initialize lists for triplet storage of columns in G
        ##
        I,V = [],[]

        ##
        ## Power balance constraints
        ##
        rp,ci,val = self.Ybus.T.CCS
        def conj(l): return [li.conjugate() for li in l]
        def smul(l): return [-li for li in l]
        def jmul(l):
           j = complex(0.0,1.0)
           return [j*li for li in l]

        def Yijv(k):
            jj = list(ci[rp[k]:rp[k+1]])
            ii = len(jj)*[k]
            vv = list(0.5*val[rp[k]:rp[k+1]])
            t = jj.index(k)
            Iret = t*[k]+jj+(len(jj)-1-t)*[k]
            Jret = jj[:t]+len(jj)*[k]+jj[t+1:]
            Vret = vv[:t] + conj(vv[:t]) + [2.0*vv[t].real] + conj(vv[t+1:]) + vv[t+1:]
            Vbret = vv[:t] + smul(conj(vv[:t])) + [complex(0.0,2.0*vv[t].imag)] + smul(conj(vv[t+1:])) + vv[t+1:]
            return Iret,Jret,Vret,jmul(Vbret)

        for k, bus in enumerate(self.busses):

            YI,YJ,YV,YbV = Yijv(k)

            if bus['id'] in self.bus_id_to_genlist:
                genlist = [self.generators[i] for i in self.bus_id_to_genlist[bus['id']]]

                L = [offset['X'] + ii + jj*Nx for ii,jj in zip(YI,YJ)]
                I.append([offset['wpl']+gen['pslack'] for gen in genlist if gen['pslack'] is not None] + L)
                V.append(len([1 for gen in genlist if gen['pslack'] is not None])*[-1.0] + YV)
                c.append(bus['Pd'] - sum([gen['Pmin'] for gen in genlist]))

                L = [offset['X'] + ii + jj*Nx for ii,jj in zip(YI,YJ)]
                I.append([offset['wql']+gen['qslack'] for gen in genlist if gen['qslack'] is not None] + L)
                V.append(len([1 for gen in genlist if gen['qslack'] is not None])*[-1.0] + YbV)
                c.append(bus['Qd'] - sum([gen['Qmin'] for gen in genlist]))

            else:
                L = [offset['X'] + ii + jj*Nx for ii,jj in zip(YI,YJ)]
                I.append(L)
                V.append(YV)
                c.append(bus['Pd'])

                L = [offset['X'] + ii + jj*Nx for ii,jj in zip(YI,YJ)]
                I.append(L)
                V.append(YbV)
                c.append(bus['Qd'])

        ##
        ## Power generation limits
        ##
        for k, gen in enumerate(self.generators_with_var_real_power()):
            I.append([offset['wpl']+k, offset['wpu']+k])
            V.append([1.0, 1.0])
            c.append(gen['Pmin'] - gen['Pmax'])

        for k, gen in enumerate(self.generators_with_var_reactive_power()):
            I.append([offset['wql']+k, offset['wqu']+k])
            V.append([1.0, 1.0])
            c.append(gen['Qmin'] - gen['Qmax'])

        ##
        ## Voltage constraints
        ##
        for k, bus in enumerate(self.busses):
            I.append([offset['ul']+k, offset['X']+k*(self.nbus+1)])
            V.append([1.0, -1.0])
            c.append(bus['minVm']**2)
        for k, bus in enumerate(self.busses):
            I.append([offset['uu']+k, offset['X']+k*(self.nbus+1)])
            V.append([1.0, 1.0])
            c.append(-bus['maxVm']**2)

        ##
        ## Line constraints
        ##
        rpf,cif,valf = self.Yf.T.CCS
        rpt,cit,valt = self.Yt.T.CCS
        def Tijv(k, fr, to, flow = 'ft'):
            if flow == 'ft':
                y = self.Ybr[k][0]
                if fr > to:
                   y.reverse()
                   fr,to = to,fr
                   flow = 'tf'
            elif flow == 'tf':
                y = self.Ybr[k][1]
                if fr > to:
                   y.reverse()
                   fr,to = to,fr
                   flow = 'ft'
            if flow == 'ft':
               Iret = [fr,to,fr]
               Jret = [fr,fr,to]
               Tret = [y[0].real, 0.5*y[1].conjugate(), 0.5*y[1]]
               Tbret = [-y[0].imag, -complex(0.0,0.5)*y[1].conjugate(), complex(0.0,0.5)*y[1]]
            elif flow == 'tf':
               Iret = [to,fr,to]
               Jret = [fr,to,to]
               Tret = [0.5*y[0],0.5*y[0].conjugate(),y[1].real]
               Tbret = [complex(0.0,0.5)*y[0],-complex(0.0,0.5)*y[0].conjugate(),-y[1].imag]
            return Iret,Jret,Tret,Tbret

        for kk,kbranch in enumerate(self.branches_with_flow_constraints()):
            k,branch = kbranch

            fr = self.bus_id_to_index[branch['from']]
            to = self.bus_id_to_index[branch['to']]

            # flow at "from" end
            Ir,Jr,Vr,Vbr = Tijv(k,fr,to,'ft')

            I.append([offset['z']+6*kk])
            V.append([1.0])
            c.append(-branch['rateA']/self.baseMVA)

            I.append([offset['z']+6*kk+1] + [offset['X'] + ii + jj*Nx for ii,jj in zip(Ir,Jr)])
            V.append([-1.0] + Vr)
            c.append(0.0)

            I.append([offset['z']+6*kk+2] + [offset['X'] + ii + jj*Nx for ii,jj in zip(Ir,Jr)])
            V.append([-1.0] + Vbr)
            c.append(0.0)

            # flow at "to" end
            Ir,Jr,Vr,Vbr = Tijv(k,fr,to,'tf')

            I.append([offset['z']+6*kk+3])
            V.append([1.0])
            c.append(-branch['rateA']/self.baseMVA)

            I.append([offset['z']+6*kk+4] + [offset['X'] + ii + jj*Nx for ii,jj in zip(Ir,Jr)])
            V.append([-1.0] + Vr)
            c.append(0.0)

            I.append([offset['z']+6*kk+5] + [offset['X'] + ii + jj*Nx for ii,jj in zip(Ir,Jr)])
            V.append([-1.0] + Vbr)
            c.append(0.0)

        ##
        ## Phase angle difference constraints
        ##
        for kk,kbranch in enumerate(self.branches_with_pad_constraints()):
            k,branch = kbranch
            tan_amin = tan(branch['angle_min']*pi/180.0)
            f = self.bus_id_to_index[branch['from']]
            t = self.bus_id_to_index[branch['to']]
            I.append([offset['lpad']+kk] + [offset['X']+f+t*Nx,offset['X']+t+f*Nx])
            V.append([1.0] + [complex(0.5*tan_amin,0.5), complex(0.5*tan_amin,-0.5)])
            c.append(0.0)
        for kk,kbranch in enumerate(self.branches_with_pad_constraints()):
            k,branch = kbranch
            tan_amax = tan(branch['angle_max']*pi/180.0)
            f = self.bus_id_to_index[branch['from']]
            t = self.bus_id_to_index[branch['to']]
            I.append([offset['upad']+kk] + [offset['X']+f+t*Nx,offset['X']+t+f*Nx])
            V.append([1.0] + [-complex(0.5*tan_amax,0.5), -complex(0.5*tan_amax,-0.5)])
            c.append(0.0)

        ##
        ## Quadratic cost -- epigraph
        ##
        for k, gen in enumerate(self.generators_with_var_real_power_and_quadratic_cost()):

            assert gen['Pcost']['ncoef'] == 3, "only quadratic cost implemented"
            assert gen['Pcost']['model'] == 2, "only quadratic cost implemented"
            ak = gen['Pcost']['coef'][-3]*self.baseMVA

            I.append([offset['t']+k,  offset['w']+3*k])
            V.append([1.0, -1.0])
            c.append(0.5)

            I.append([offset['t']+k,  offset['w']+3*k+1])
            V.append([-1.0, -1.0])
            c.append(0.5)

            I.append([offset['wpl']+gen['pslack'], offset['w']+3*k+2])
            V.append([sqrt(2.0*ak), -1.0])
            c.append(0.0)

        ##
        ## Fixed cost
        ##
        self.const_cost = 0.0
        for kk,gen in enumerate(self.generators):
            if gen['Pcost']['ncoef'] == 2 or (gen['Pcost']['ncoef'] == 3 and gen['Pcost']['coef'][0] == 0.0):
                self.const_cost += gen['Pcost']['coef'][-1]/self.baseMVA + gen['Pcost']['coef'][-2]*gen['Pmin']
            elif gen['Pcost']['ncoef'] == 3:
                self.const_cost += gen['Pcost']['coef'][-1]/self.baseMVA \
                  + gen['Pcost']['coef'][-2]*gen['Pmin'] \
                  + self.baseMVA*gen['Pcost']['coef'][-3]*gen['Pmin']**2
        self.const_cost *= self.baseMVA

        ##
        ## Build c, h, and G
        ##
        c = matrix(c)
        J = [len(Ii)*[j] for j,Ii in enumerate(I)]
        G = spmatrix([v for v in chain(*V)],
                     [v for v in chain(*I)],
                     [v for v in chain(*J)],(N,len(c)))

        self.problem_data = (c, G, h, dims)
        return

    def solve(self, solver = "mosek"):
        if self.to_real == False: raise ValueError("Solvers do not support complex-valued data.")
        sol = {}
        c,G,h,dims = self.problem_data
        if solver == "mosek":
            if self.__verbose:
               msk.options[msk.mosek.iparam.log] = 1
            else:
               msk.options[msk.mosek.iparam.log] = 0
            solsta,mu,zz = msk.conelp(c,G,matrix(h),dims)
            sol['status'] = str(solsta).split('.')[-1]
        elif solver == "cvxopt":
            if self.__verbose:
                options = {'show_progress':True}
            else:
                options = {'show_progress':False}
            csol = solvers.conelp(c,G,matrix(h),dims,options=options)
            zz = csol['z']
            mu = csol['x']
            sol['status'] = csol['status']
        else:
            raise ValueError("Unknown solver %s" % (solver))
        if zz is None: return sol

        sol['mu'] = mu
        offset = self.offset
        sol['t'] = zz[:offset['wpl']]
        sol['wpl'] = zz[offset['wpl']:offset['wpu']]
        sol['wpu'] = zz[offset['wpu']:offset['wql']]
        sol['wql'] = zz[offset['wql']:offset['wqu']]
        sol['wqu'] = zz[offset['wqu']:offset['ul']]
        sol['ul'] = zz[offset['ul']:offset['uu']]
        sol['uu'] = zz[offset['uu']:offset['z']]
        sol['z'] = zz[offset['z']:offset['w']]
        sol['w'] = zz[offset['w']:offset['X']]

        if self.conversion:
            dims = self.problem_data[3]
            offset = dims['l'] + sum(dims['q'])
            self.Xc = []
            sol['eigratio'] = []
            for k,sk in enumerate(dims['s']):
                zk = matrix(zz[offset:offset+sk**2],(sk,sk))
                offset += sk**2
                zkr = 0.5*(zk[:sk//2,:sk//2] + zk[sk//2:,sk//2:])
                zki = 0.5*(zk[sk//2:,:sk//2] - zk[:sk//2,sk//2:])
                zki[::sk+1] = 0.0
                zk = zkr + complex(0,1.0)*zki
                self.Xc.append(zk)
                ev = matrix(0.0,(zk.size[0],1),tc='d')
                lapack.heev(+zk,ev)
                ev = sorted(list(ev),reverse=True)
                sol['eigratio'].append(ev[0]/ev[1])

            # Build partial Hermitian matrix
            z = matrix([zk[:] for zk in self.Xc])
            blki,I,J,bn = self.blocks_to_sparse[0]
            X = spmatrix(z[blki],I,J)
            idx = [i for i,ij in enumerate(zip(X.I,X.J)) if ij[0] > ij[1]]
            sol['X'] = chompack.tril(X) + spmatrix(X.V[idx].H, X.J[idx], X.I[idx], X.size)

        else:
            X = matrix(zz[offset['X']:],(2*self.nbus,2*self.nbus))
            Xr = X[:self.nbus,:self.nbus]
            Xi = X[self.nbus:,:self.nbus]
            Xi[::self.nbus+1] = 0.0
            X = Xr + complex(0.0,1.0)*Xi
            sol['X'] = +X

            V = matrix(0.0,(self.nbus,5),tc='z')
            w = matrix(0.0,(self.nbus,1))
            lapack.heevr(X, w, Z = V, jobz='V', range='I', il = self.nbus-4, iu = self.nbus)
            sol['eigratio'] = [w[4]/w[3]]
            if w[4]/w[3] < self.eigtol and self.__verbose:
                print("Eigenvalue ratio smaller than %e."%(self.eigtol))
            sol['eigs'] = w[:5]
            V = V[:,-1]*sqrt(w[4])
            sol['V'] = V

        # Branch injections
        sol['Sf'] = self.baseMVA*(sol['z'][1::6] + complex(0.0,1.0)*sol['z'][2::6])
        sol['St'] = self.baseMVA*(sol['z'][4::6] + complex(0.0,1.0)*sol['z'][5::6])

        # Generation
        sol['Sg'] = (matrix([gen['Pmin'] for gen in self.generators]) +\
                     matrix([0.0 if gen['pslack'] is None else sol['wpl'][gen['pslack']] for gen in self.generators])) +\
                     complex(0.0,1.0)*(matrix([gen['Qmin'] for gen in self.generators]) +\
                     matrix([0.0 if gen['qslack'] is None else sol['wql'][gen['qslack']] for gen in self.generators]))
        Pg = sol['Sg'].real()
        Qg = sol['Sg'].imag()
        for k,gen in enumerate(self.generators):
            gen['Pg'] = Pg[k]
            gen['Qg'] = Qg[k]
        sol['Sg'] *= self.baseMVA

        sol['cost'] = 0.0
        for ii,gen in enumerate(self.generators):
            for nk in range(gen['Pcost']['ncoef']):
                sol['cost'] += gen['Pcost']['coef'][-1-nk]*(Pg[ii]*self.baseMVA)**nk

        sol['cost_objective'] = -(self.problem_data[0].T*mu)[0]*self.cost_scale + self.const_cost

        sol['Vm'] = sqrt(matrix(sol['X'][::self.nbus+1]).real())
        return sol

    def export(self, fname):
        """
        Save cone LP to file.
        """

        assert type(fname) is str, "fname must be a string"
        fmt = fname.split('.')[-1]
        if fmt == 'mat':
            assert __scipy__, "Export error: exporting to .mat requires scipy."
            c,G,h,dims = self.problem_data
            cost_scale = self.cost_scale
            if G.typecode == 'd':
                Gs = csr_matrix((np.array(G.V).squeeze(), (np.array(G.I).squeeze(), np.array(G.J).squeeze())), shape=np.array(G.size))
                if isinstance(h,matrix):
                    ht = np.array(h).squeeze()
                else:
                    ht = csr_matrix((np.array(h.V).squeeze(), (np.array(h.I).squeeze(), np.array(h.J).squeeze())), shape=np.array(h.size))
                mdict = {'A':Gs.transpose(),
                         'b':-np.array(c).squeeze(),
                         'c':ht,
                         'K':{'l':float(dims['l']),'q':np.array(dims['q'],dtype='<f8'),'s':np.array(dims['s'],dtype='<f8')},
                         'opfsdr':{
                           'offsetx':np.array(self.offset['X']).squeeze(),
                           'cost_scale':np.array(cost_scale),
                           'cost_const':self.const_cost,
                           'dual_offset':self.dual_offset}
                         }
                savemat(fname, mdict, format='5', do_compression=True, oned_as='column')
            else:
                print("Export of complex-valued data not yet implemented.")
        elif fmt in ['pickle','pkl','pbz2']:
            c,G,h,dims = self.problem_data
            data = {'c':c,'G':G,'h':h,
                    'dims':dims,
                    'opfsdr':{
                       'offsetx':matrix(self.offset['X'], tc='d'),
                       'cost_scale':matrix(self.cost_scale,tc='d'),
                       'cost_const':self.const_cost,
                       'dual_offset':self.dual_offset}
                    }
            if hasattr(self,"blocks_to_sparse"): data["blocks_to_sparse"] = self.blocks_to_sparse
            if fmt in ['pickle','pkl']:
                with open(fname, 'w') as f: pickle.dump(data, f)
            else:
                import bz2
                with bz2.BZ2File(fname, 'w') as f: pickle.dump(data, f)
        else:
           raise ValueError("Unknown format.")
        return

BUS_FORMAT = r"""
   Bus Data Format
       1   bus number (positive integer)
       2   bus type
               PQ bus          = 1
               PV bus          = 2
               reference bus   = 3
               isolated bus    = 4
       3   Pd, real power demand (MW)
       4   Qd, reactive power demand (MVAr)
       5   Gs, shunt conductance (MW demanded at V = 1.0 p.u.)
       6   Bs, shunt susceptance (MVAr injected at V = 1.0 p.u.)
       7   area number, (positive integer)
       8   Vm, voltage magnitude (p.u.)
       9   Va, voltage angle (degrees)
   (-)     (bus name)
       10  baseKV, base voltage (kV)
       11  zone, loss zone (positive integer)
   (+) 12  maxVm, maximum voltage magnitude (p.u.)
   (+) 13  minVm, minimum voltage magnitude (p.u.)
"""

GEN_FORMAT = r"""
   Generator Data Format
       1   bus number
   (-)     (machine identifier, 0-9, A-Z)
       2   Pg, real power output (MW)
       3   Qg, reactive power output (MVAr)
       4   Qmax, maximum reactive power output (MVAr)
       5   Qmin, minimum reactive power output (MVAr)
       6   Vg, voltage magnitude setpoint (p.u.)
   (-)     (remote controlled bus index)
       7   mBase, total MVA base of this machine, defaults to baseMVA
   (-)     (machine impedance, p.u. on mBase)
   (-)     (step up transformer impedance, p.u. on mBase)
   (-)     (step up transformer off nominal turns ratio)
       8   status,  >  0 - machine in service
                    <= 0 - machine out of service
   (-)     (% of total VAr's to come from this gen in order to hold V at
               remote bus controlled by several generators)
       9   Pmax, maximum real power output (MW)
       10  Pmin, minimum real power output (MW)
   (2) 11  Pc1, lower real power output of PQ capability curve (MW)
   (2) 12  Pc2, upper real power output of PQ capability curve (MW)
   (2) 13  Qc1min, minimum reactive power output at Pc1 (MVAr)
   (2) 14  Qc1max, maximum reactive power output at Pc1 (MVAr)
   (2) 15  Qc2min, minimum reactive power output at Pc2 (MVAr)
   (2) 16  Qc2max, maximum reactive power output at Pc2 (MVAr)
   (2) 17  ramp rate for load following/AGC (MW/min)
   (2) 18  ramp rate for 10 minute reserves (MW)
   (2) 19  ramp rate for 30 minute reserves (MW)
   (2) 20  ramp rate for reactive power (2 sec timescale) (MVAr/min)
   (2) 21  APF, area participation factor
"""

BRANCH_FORMAT = r"""
   Branch Data Format
       1   f, from bus number
       2   t, to bus number
   (-)     (circuit identifier)
       3   r, resistance (p.u.)
       4   x, reactance (p.u.)
       5   b, total line charging susceptance (p.u.)
       6   rateA, MVA rating A (long term rating)
       7   rateB, MVA rating B (short term rating)
       8   rateC, MVA rating C (emergency rating)
       9   ratio, transformer off nominal turns ratio ( = 0 for lines )
           (taps at 'from' bus, impedance at 'to' bus,
            i.e. if r = x = 0, then ratio = Vf / Vt)
       10  angle, transformer phase shift angle (degrees), positive => delay
   (-)     (Gf, shunt conductance at from bus p.u.)
   (-)     (Bf, shunt susceptance at from bus p.u.)
   (-)     (Gt, shunt conductance at to bus p.u.)
   (-)     (Bt, shunt susceptance at to bus p.u.)
       11  initial branch status, 1 - in service, 0 - out of service
   (2) 12  minimum angle difference, angle(Vf) - angle(Vt) (degrees)
   (2) 13  maximum angle difference, angle(Vf) - angle(Vt) (degrees)
"""

COST_FORMAT = r"""
   Generator Cost Data Format
       NOTE: If gen has ng rows, then the first ng rows of gencost contain
       the cost for active power produced by the corresponding generators.
       If gencost has 2*ng rows then rows ng+1 to 2*ng contain the reactive
       power costs in the same format.
       1   model, 1 - piecewise linear, 2 - polynomial
       2   startup, startup cost in US dollars
       3   shutdown, shutdown cost in US dollars
       4   N, number of cost coefficients to follow for polynomial
           cost function, or number of data points for piecewise linear
       5 and following, parameters defining total cost function f(p),
           units of f and p are $/hr and MW (or MVAr), respectively.
           (MODEL = 1) : p0, f0, p1, f1, ..., pn, fn
               where p0 < p1 < ... < pn and the cost f(p) is defined by
               the coordinates (p0,f0), (p1,f1), ..., (pn,fn) of the
               end/break-points of the piecewise linear cost function
           (MODEL = 2) : cn, ..., c1, c0
               n+1 coefficients of an n-th order polynomial cost function,
               starting with highest order, where cost is
               f(p) = cn*p^n + ... + c1*p + c0
"""

BUS_TYPE =  {1: 'PQ bus',
             2: 'PV bus',
             3: 'reference bus',
             4: 'isolated bus'}
